import os
import argparse
import logging
import warnings 
import json 
import math
import model as instruct_model
import datasets
import torch
import torchvision
from torch.utils.data import dataloader
import utils
from tqdm import tqdm
import random
import numpy as np
from omegaconf import OmegaConf
from lavis.models import load_preprocess
from torch.cuda.amp import autocast as autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from datetime import datetime
warnings.filterwarnings("ignore")
torch.set_num_threads(8)

parser = argparse.ArgumentParser()


parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--eval_frequency', type=int, default=1)
parser.add_argument('--early_stop_num', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--img_size', type=int, default=224)

parser.add_argument('--model_dir', default=None, help="save results")
parser.add_argument('--model_name', default=None, help="save results")
parser.add_argument('--gt_dir', default='./eval_data', help='the ground-truth caption')
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--dataset_json_path', default='webqa_data/webqa_train2.json', help="dataset_json_path")
parser.add_argument('--image_data_path', default='webqa_data/imageid_train.csv', help="image_data_path")
parser.add_argument('--image_data_idx', default='webqa_data/imageid_train.csv', help="image_data_idx")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_step = 0
def get_dataset(args):
    cfg = OmegaConf.load('lavis/configs/models/blip2/blip2_instruct_flant5xl.yaml')
    img_preprocess, _ = load_preprocess(cfg.preprocess)
    
    val_set = datasets.WebqaDataset(dataset_json_path=args.dataset_json_path,
                                    image_data_path=args.image_data_path,
                                    lineidx=args.image_data_idx,
                                    transform=img_preprocess['eval'],
                                    split='test')
    return val_set

def create_model(model_path):
    
    model = instruct_model.Blip2T5InstructEnhance.load_pretrained_model(model_type='flant5xl')


    model.load_state_dict(torch.load(model_path))


    model.to(device)


    total_params = sum([param.nelement() for param in model.parameters()])
    print('Total Params: {:.3f}M'.format(total_params / 1000000))


    print('======= MODULE PRINT ========')
    for n, c in model.named_children():
        print("[children module: {}]".format(n))
        for cn, p in c.named_parameters():
            print("  >>  " + n + '.' + cn, "|", p.size(), "|", p.requires_grad, "|", p.device)
    print('=============================')
    
    return model


def eval(args,model, val_set):
    model.eval()
    
    val_dataloader = dataloader.DataLoader(val_set, 
                                            shuffle=False, 
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=False)

    generate_results = {}

    with torch.no_grad():
        for data in tqdm(val_dataloader):
            query_ids = data['query_id']
            questions = data['Question']
            image_data = data['image'].to(device)
            data['image'] = image_data
            answers = model.generate(data)
            for qid, question, answer in zip(query_ids, questions, answers):
                generate_results[qid]={'Q':question,'A':answer}
    
    json.dump(generate_results, open(os.path.join(args.model_dir,'result.json'), 'w'),indent=4)
    
    ### compute test metrics

if __name__ == '__main__':
    
    print(args)
    
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    print('Loading the datasets and model...')

    model_path=os.path.join(args.model_dir,args.model_name)
    val_set = get_dataset(args)
    model= create_model(model_path)
    start_time = datetime.now()

    eval(args,model,val_set)
    