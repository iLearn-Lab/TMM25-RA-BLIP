import os
from PIL import Image
from lavis.models import load_model_and_preprocess
from io import BytesIO
import base64, json
import torch
import re
from lavis.models import load_preprocess 
from omegaconf import OmegaConf
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import sys
class WebqaDataset(Dataset):
    def __init__(self, dataset_json_path, image_data_path, lineidx=None, transform='', split='train') -> None:
        super().__init__()
        with open(dataset_json_path, 'r') as f:
            self.caption_data = json.load(f)
        self.caption_ids = list(self.caption_data.keys())
        if lineidx:
            with open(lineidx, "r") as fp_lineidx:
                self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        else:
            self.lineidx=lineidx
        self.split = split
        self.transform = transform
        #print(self.transform)
        self.image_data_path = image_data_path
        self.text_processor = BlipCaptionProcessor()
    def __len__(self):
        return len(self.caption_data)
    
    def __getitem__(self, index):
        # print(self.caption_ids[index])
        item = self.caption_data[self.caption_ids[index]]
        query_id = self.caption_ids[index]
        #print(query_id)
        Q = item['Q']
        A = item['A'][0]

        img_pos_ids = []
        img_pos_caption = []
        for pos_img in item['img_posFacts']:
            img_pos_ids += [pos_img['image_id']]
            img_pos_caption += [pos_img['caption']]

        txt_pos_title = []
        txt_pos_fact = []
        for pos_txt in item['txt_posFacts']:
            txt_pos_title += [pos_txt['title']]
            txt_pos_fact += [pos_txt['fact']]

        out = {}
        image_data, image_caption, valid_image_num = self.get_image(img_pos_ids, img_pos_caption, k=2)
        pos_text = self.get_text(txt_pos_title, txt_pos_fact)
        question = self.text_processor(Q)

        if self.split == 'train':
            out['query_id'] = query_id
            out['image'] = image_data
            out['valid_image_num'] = valid_image_num
            out['prompt'] = " | ".join(filter(lambda x: len(x) != 0,[image_caption, pos_text, question]))
            out['Answer'] = self.text_processor(A)
            out['Question'] = self.text_processor(Q)
        else :
            out['query_id'] = query_id
            out['image'] = image_data
            out['valid_image_num'] = valid_image_num
            out['prompt'] = " | ".join(filter(lambda x: len(x) != 0,[image_caption, pos_text, question]))
            out['Question'] = self.text_processor(Q)
        return out

    def get_image(self, image_ids,image_caption, k=2):
        #print(image_ids)
        if len(image_ids) == 0:
            return torch.stack([torch.zeros((3, 224, 224)), torch.zeros((3, 224, 224))]), "", 0
        else:
            image_data = []
            if self.lineidx is None:
                with open(self.image_data_path,'r') as f1:
                    dataset_J1 = json.load(f1)         
                for image_id in image_ids[:k]:
                    #print(dataset_J1[str(image_id)])
                    image_data += [self.transform(Image.open(BytesIO(base64.b64decode(dataset_J1[str(image_id)]))).convert('RGB'))]
            else:
                with open(self.image_data_path, "r") as fp:
                    for image_id in image_ids[:k]:
                        fp.seek(self.lineidx[int(image_id) % 10000000])
                        img_id, img_data = fp.readline().strip().split('\t')
                        assert int(img_id) == int(image_id), 'image_id is different from img_id'
                        image_data += [self.transform(Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB'))]
            image_caption = ";".join(["caption of image %s is that %s" % (i+1,s) for i,s in enumerate(image_caption)])
        ### pad zero image tensor
        if len(image_ids) < k:
            extended_num = k - len(image_ids)
            image_data += [torch.zeros((3, 224, 224))] * extended_num
        #print(image_data[0].shape) torch.Size([3, 224, 224])
        #print(len(image_data)) 2
        #print(torch.stack(image_data).shape) torch.Size([2, 3, 224, 224])
        return torch.stack(image_data), image_caption, len(image_ids)    
            
    def get_text(self, text_title, text_fact):
        if len(text_title) == 0:
            return ""
        else:
            text = []
            for title, fact in zip(text_title, text_fact):
                t = "Title:%s, Fact:%s" % (self.text_processor(title), self.text_processor(fact))
                text.append(t)
            text = ";".join(text)
            return text
        

class WebqaEnhancedTrainDataset(WebqaDataset):
    def __init__(self, dataset_json_path, image_data_path, lineidx=None, transform='', split='train') -> None:
        super().__init__(
            dataset_json_path=dataset_json_path,
            image_data_path=image_data_path,
            lineidx=lineidx,
            transform=transform,
            split=split,
        )
        print('We would like to request your feedback on this question is relevant to which of the following references. Relevance refers to the degree to which the reference can answer the question. \nThe input format is Question: content, Reference [number]: content \nThe Output format is: Related content is [number]\n')

    def __getitem__(self, index):
        empty_img = 'iVBORw0KGgoAAAANSUhEUgAAAwAAAAMAAQAAAAC7+j0jAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QAAd2KE6QAAAAHdElNRQfhCAgMOwDRoBf1AAABFklEQVR42u3NIQEAAAACIP+f1hU2OEB6FoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEgq8BaxHwwz4L1VsAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTctMDgtMDhUMTI6NTk6MDArMDA6MDAVAzDwAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE3LTA4LTA4VDEyOjU5OjAwKzAwOjAwZF6ITAAAAABJRU5ErkJggg=='
        item = self.caption_data[self.caption_ids[index]]
        query_id = self.caption_ids[index]
        Q = item['Q']
        A = item['A'][0]

        img_pos_ids = []
        img_pos_caption = []
        pos = 'img'
        for pos_img in item['img_posFacts']:
            img_pos_ids += [pos_img['image_id']]
            img_pos_caption += [pos_img['caption']]

        if len(item['img_negFacts']) == 0:
            img_neg_ids = 300
            img_neg_caption = empty_img
            image_neg_data = self.transform(Image.open(BytesIO(base64.b64decode(empty_img))).convert('RGB'))
        else:
            for neg_img in item['img_negFacts']:
                img_neg_ids = neg_img['image_id']
                img_neg_caption = neg_img['caption']
                image_neg_data = self.get_image2(img_neg_ids)
                break

        txt_pos_ids = []
        txt_pos_title = []
        txt_pos_fact = []
        for pos_txt in item['txt_posFacts']:
            txt_pos_ids += [pos_txt['snippet_id']]
            txt_pos_title += [pos_txt['title']]
            txt_pos_fact += [pos_txt['fact']]
            pos = 'txt'

        if len(item['txt_negFacts']) == 0:
            txt_neg_ids = "ddd"
            txt_neg_title = ["empty"]
            txt_neg_fact = ["empty"]
        else:
            for neg_txt in item['txt_negFacts']:
                txt_neg_ids = neg_txt['snippet_id']
                txt_neg_title = [neg_txt['title']]
                txt_neg_fact = [neg_txt['fact']]
                break

        pos_text = self.get_text(txt_pos_title, txt_pos_fact)
        neg_text = self.get_text(txt_neg_title, txt_neg_fact)
        out = {}
        retrieval = {}

        image_data, image_caption, valid_image_num = self.get_image(img_pos_ids, img_pos_caption, k=2)
        question = self.text_processor(Q)
        retrieval['pos'] = pos

        prompt = 'We would like to request your feedback on this question is relevant to which of the following references. Relevance refers to the degree to which the reference can answer the question. \nThe input format is Question: content, Reference [number]: content \nThe Output format is: Related content is [number]\n'
        prompt = prompt + "Question: \"" + self.text_processor(Q) + '\"\n' + "The references are as follows: \n"

        if pos == 'img':
            retrieval['ids'] = str(img_pos_ids[0])
            retrieval['pos_text'] = img_pos_caption[0]
            retrieval['image_pos_data'] = image_data[0]

            prompt = prompt + "Reference [" + retrieval['ids'] + "]: " + "The reference content is in image format, the image caption is \"" + retrieval['pos_text'] + '\"\n'

            retrieval['neg_ids'] = txt_neg_ids
            retrieval['neg_text'] = neg_text
            retrieval['image_neg_data'] = self.transform(Image.open(BytesIO(base64.b64decode(empty_img))).convert('RGB'))

            prompt = prompt + "Reference [" + retrieval['neg_ids'] + "]: " + "The reference content is in text format, the text content is \"" + retrieval['neg_text'] + '\"\n'

            answer = "The relevant reference is Reference [" + retrieval['ids'] + "]"
            instruction = "caption of image is " + retrieval['pos_text'] + " | question is " + question
        else:
            pos_text_1 = self.get_text([txt_pos_title[0]], [txt_pos_fact[0]])
            retrieval['ids'] = txt_pos_ids[0]
            retrieval['pos_text'] = pos_text_1
            retrieval['image_pos_data'] = self.transform(Image.open(BytesIO(base64.b64decode(empty_img))).convert('RGB'))

            prompt = prompt + "Reference [" + retrieval['ids'] + "]: " + "The reference content is in text format, the text content is \"" + retrieval['pos_text'] + '\"\n'

            retrieval['neg_ids'] = str(img_neg_ids)
            retrieval['neg_text'] = img_neg_caption
            retrieval['image_neg_data'] = image_neg_data

            prompt = prompt + "Reference [" + retrieval['neg_ids'] + "]: " + "The reference content is in image format, the image caption is \"" + retrieval['neg_text'] + '\"\n'

            answer = "The relevant reference is Reference [" + retrieval['ids'] + "]."
            instruction = "caption of image is " + retrieval['neg_text'] + " | question is " + question

        retrieval['prompt'] = prompt
        retrieval['answer'] = answer
        retrieval['instruction'] = instruction

        if self.split == 'train':
            out['query_id'] = query_id
            out['image'] = image_data
            out['valid_image_num'] = valid_image_num
            out['prompt'] = " | ".join(filter(lambda x: len(x) != 0, [image_caption, pos_text, question]))
            out['Answer'] = self.text_processor(A)
            out['Question'] = self.text_processor(Q)
        else:
            out['query_id'] = query_id
            out['image'] = image_data
            out['valid_image_num'] = valid_image_num
            out['prompt'] = " | ".join(filter(lambda x: len(x) != 0, [image_caption, pos_text, question]))
            out['Question'] = self.text_processor(Q)

        return out, retrieval

    def get_image2(self, image_id):
        if self.lineidx is None:
            with open(self.image_data_path, 'r') as f1:
                dataset_J1 = json.load(f1)
                image_data = self.transform(Image.open(BytesIO(base64.b64decode(dataset_J1[str(image_id)]))).convert('RGB'))
        else:
            with open(self.image_data_path, "r") as fp:
                fp.seek(self.lineidx[int(image_id) % 10000000])
                img_id, img_data = fp.readline().strip().split('\t')
                assert int(img_id) == int(image_id), 'image_id is different from img_id'
                image_data = self.transform(Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB'))

        return image_data





    
class BlipCaptionProcessor(object):
    def __init__(self, prompt="", max_words=256):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = " ".join([self.prompt, self.pre_caption(caption)])

        return caption

    def pre_caption(self, caption):
        caption = re.sub(
            r"[\"*#:;~]",
            " ",
            #caption.lower(),
            caption,
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption



        

    
