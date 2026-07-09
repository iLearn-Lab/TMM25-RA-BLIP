from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers.modeling_outputs import BaseModelOutput
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast,AutoTokenizer
import torch
import torch.nn as nn
import random
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn import functional as F
from lavis.models.base_model import all_gather_with_grad, concat_all_gather


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config,num_labels=2):
        super().__init__()
        classifier_dropout=0
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Blip2T5Instruct(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=512,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        if vit_model == 'clip_L':
            num_patchs = 2 * (self.visual_encoder.num_patches + 1)
        elif vit_model == 'eva_clip_g':
            num_patchs = 2 * (self.visual_encoder.patch_embed.num_patches + 1)
        visual_num_features = self.visual_encoder.num_features
        self.pos_embeds = nn.Parameter(torch.zeros(1, num_patchs, visual_num_features))
        self.type_embeds = nn.Parameter(torch.zeros(2, 1, 1, visual_num_features))

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        print("max_txt_len: ",max_txt_len)
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def forward(self, input_data):
        with self.maybe_autocast():
            image1_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 0, :]))
            image2_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 1, :]))
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)


        for i, view_img_num in enumerate(input_data['valid_image_num']):
            if view_img_num == 1:
                image2_atts[i, :] = 0

            elif view_img_num == 0:
                image1_atts[i, :] = 0
                image2_atts[i, :] = 0

        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # B 32 768
        ## input_data.text: image_caption(if have) + pos_text + question 
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                input_data['prompt'],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)

        ## input_data.text: image_caption(if have) + pos_text + question
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                input_data['prompt'],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            output_tokens = self.t5_output_tokenizer(
                input_data['Answer'],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return loss
    
    @torch.no_grad()
    def generate(
        self,
        input_data,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            image1_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 0, :]))
            image2_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 1, :]))
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        if self.pos_embeds is not None:
            image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)


        for i, view_img_num in enumerate(input_data['valid_image_num']):
            if view_img_num == 1:
                image2_atts[i, :] = 0

            elif view_img_num == 0:
                image1_atts[i, :] = 0
                image2_atts[i, :] = 0


        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        ### input_data.text: img_cap + pos_text + question
        text_Qformer = self.tokenizer(
            input_data['prompt'],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)
        input_tokens = self.t5_tokenizer(
            input_data['prompt'],
            padding="longest",
            return_tensors="pt"
        ).to(image_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    @classmethod
    def load_pretrained_model(cls, model_type):
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        
        vit_model = model_cfg.get("vit_model", "eva_clip_g")
        img_size = model_cfg.get("image_size")
        num_query_token = model_cfg.get("num_query_token")
        t5_model = model_cfg.get("t5_model")

        drop_path_rate = model_cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = model_cfg.get("use_grad_checkpoint", False)
        vit_precision = model_cfg.get("vit_precision", "fp16")
        freeze_vit = model_cfg.get("freeze_vit", True)



        prompt = model_cfg.get("prompt", "")
        max_txt_len = model_cfg.get("max_txt_len", 512)
        max_output_txt_len = model_cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = model_cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = model_cfg.get("num_few_shot_examples", 0)
        few_shot_prob = model_cfg.get("few_shot_prob", 0.0)

        qformer_text_input = model_cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
        )
        print("model_cfg: ", model_cfg)
        model.load_checkpoint_from_config(model_cfg)

        return model
    

class Blip2T5InstructEnhance(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=512,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        if vit_model == 'clip_L':
            num_patchs = 2 * (self.visual_encoder.num_patches + 1)
        elif vit_model == 'eva_clip_g':
            num_patchs = 2 * (self.visual_encoder.patch_embed.num_patches + 1)
        visual_num_features = self.visual_encoder.num_features
        self.pos_embeds = nn.Parameter(torch.zeros(1, num_patchs, visual_num_features))
        self.type_embeds = nn.Parameter(torch.zeros(2, 1, 1, visual_num_features))

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        print("max_txt_len: ",max_txt_len)
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def forward(self, input_data,retrieval):
        with self.maybe_autocast():

            image1_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 0, :]))
            image2_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 1, :]))
        
        # 推理方面
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        for i, view_img_num in enumerate(input_data['valid_image_num']):
            if view_img_num == 1:
                image2_atts[i, :] = 0

            elif view_img_num == 0:
                image1_atts[i, :] = 0
                image2_atts[i, :] = 0

        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                input_data['prompt'],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)

        ## input_data.text: image_caption(if have) + pos_text + question
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                input_data['prompt'],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            output_tokens = self.t5_output_tokenizer(
                input_data['Answer'],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss1 = outputs.loss


        image1_embeds = self.ln_vision(self.visual_encoder(retrieval['image_pos_data']))
        image2_embeds = self.ln_vision(self.visual_encoder(retrieval['image_neg_data']))
        
        
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        

        for i, ids in enumerate(retrieval['ids']):
            #print(ids)
            if str(ids)[0] == "d":

                image1_atts[i, :] = 0
            
            else:

                image2_atts[i, :] = 0


        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)


        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                retrieval['instruction'],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
            
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                retrieval['prompt'],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            output_tokens = self.t5_output_tokenizer(
                retrieval['answer'],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss2 = outputs.loss
            
            loss = loss1 + 0.1*loss2
            
            return loss
    
    @torch.no_grad()
    def generate(
        self,
        input_data,
        use_nucleus_sampling=False,
        num_beams=5, # 5
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            image1_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 0, :]))
            image2_embeds = self.ln_vision(self.visual_encoder(input_data['image'][:, 1, :]))
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        if self.pos_embeds is not None:
            image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)


        for i, view_img_num in enumerate(input_data['valid_image_num']):
            if view_img_num == 1:
                image2_atts[i, :] = 0

            elif view_img_num == 0:
                image1_atts[i, :] = 0
                image2_atts[i, :] = 0


        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        ### input_data.text: img_cap + pos_text + question
        text_Qformer = self.tokenizer(
            input_data['prompt'],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)

        input_tokens = self.t5_tokenizer(
            input_data['prompt'],
            padding="longest",
            return_tensors="pt"
        ).to(image_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text
    
    @torch.no_grad()
    def generate_case(
        self,
        retrieval,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            image1_embeds = self.ln_vision(self.visual_encoder(retrieval['image_pos_data']))
            image2_embeds = self.ln_vision(self.visual_encoder(retrieval['image_neg_data']))
        if self.type_embeds is not None:
            image1_embeds = image1_embeds + self.type_embeds[0]
            image2_embeds = image2_embeds + self.type_embeds[1]
        image_embeds = torch.cat([image1_embeds, image2_embeds], dim=1)
        if self.pos_embeds is not None:
            image_embeds = image_embeds + self.pos_embeds

        image1_atts = torch.ones(image1_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        for i, ids in enumerate(retrieval['ids']):
            print(ids)
            if str(ids)[0] == "d":
                print('img1')
                image1_atts[i, :] = 0
            

            else:
                print('img2')
                image2_atts[i, :] = 0


        image_atts = torch.cat([image1_atts, image2_atts], dim=-1)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        ### input_data.text: img_cap + pos_text + question
        text_Qformer = self.tokenizer(
            retrieval['instruction'],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image_embeds.device)
        # 同样的输入再用t5编码，输入到t5-encoder中
        input_tokens = self.t5_tokenizer(
            retrieval['prompt'],
            padding="longest",
            return_tensors="pt"
        ).to(image_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    @classmethod
    def load_pretrained_model(cls, model_type):
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        
        vit_model = model_cfg.get("vit_model", "eva_clip_g")
        img_size = model_cfg.get("image_size")
        num_query_token = model_cfg.get("num_query_token")
        t5_model = model_cfg.get("t5_model")

        drop_path_rate = model_cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = model_cfg.get("use_grad_checkpoint", False)
        vit_precision = model_cfg.get("vit_precision", "fp16")
        freeze_vit = model_cfg.get("freeze_vit", True)



        prompt = model_cfg.get("prompt", "")
        max_txt_len = model_cfg.get("max_txt_len", 256) # 512
        max_output_txt_len = model_cfg.get("max_output_txt_len", 128) # 256

        apply_lemmatizer = model_cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = model_cfg.get("num_few_shot_examples", 0)
        few_shot_prob = model_cfg.get("few_shot_prob", 0.0)

        qformer_text_input = model_cfg.get("qformer_text_input", True)

        print("t5_model: ",t5_model)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
        )
        print("model_cfg: ", model_cfg)
        model.load_checkpoint_from_config(model_cfg)

        return model
    
