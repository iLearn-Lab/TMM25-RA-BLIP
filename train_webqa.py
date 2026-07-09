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
import shutil  
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from datetime import datetime
import time
from lavis.common.optims import LinearWarmupCosineLRScheduler
warnings.filterwarnings("ignore")
torch.set_num_threads(8)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))

parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--warm_up_steps', type=int, default=1000)
parser.add_argument('--eval_frequency', type=int, default=1)
parser.add_argument('--early_stop_num', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--init_lr', type=float, default=1e-7)
parser.add_argument('--min_lr', type=float, default=5e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--img_size', type=int, default=364)
parser.add_argument('--model_dir', default='exp731/finetune', help="save results")
parser.add_argument('--gt_dir', default='./eval_data', help='the ground-truth caption')
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--dataset_json_path', default='webqa_data/webqa_train2.json', help="dataset_json_path")
parser.add_argument('--image_data_path', default='webqa_data/imageid_train.csv', help="image_data_path")
parser.add_argument('--image_data_idx', default=None, help="image_data_idx")

parser.add_argument('--model_type', default='flant5xl', help="image_data_path")
parser.add_argument('--yaml', default='lavis/configs/models/blip2/blip2_instruct_flant5xl.yaml', help="image_data_path")
args = parser.parse_args()


def init_dis_mode(args):
    try:
        ntry = 5
        rng = random.Random(234)
        rnd_delay = rng.random()
        for k in range(ntry):
            try:
                dist.init_process_group(backend="nccl")
                break
            except RuntimeError as e:
                if k < ntry:
                    nseconds = rnd_delay * (k + 1)
                    print("init_process_group failed: {}. Retry {}/{} in {:.2f}s.".format(
                        e, k+1, ntry, nseconds))
                    time.sleep(nseconds)
                    continue
                else:
                    raise

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.distributed = args.world_size > 1
        args.gpu = args.rank % torch.cuda.device_count()
        print(torch.cuda.device_count(), args.gpu, args.rank, args.world_size, args.distributed)
    except (AttributeError, ValueError) as e:
        print("distributed is not enabled")
        print(e)
        args.rank = 0
        args.world_size = 1
        args.distributed = False
        args.gpu = args.rank % torch.cuda.device_count()
    
    torch.cuda.set_device(args.gpu)
    if args.distributed:
        torch.distributed.barrier()


global_step = 0


def get_dataset(args):
    cfg = OmegaConf.load(args.yaml)
    img_preprocess, _ = load_preprocess(cfg.preprocess)

    train_set = datasets.WebqaEnhancedTrainDataset(dataset_json_path=args.dataset_json_path,
                                    image_data_path=args.image_data_path,
                                    lineidx=args.image_data_idx,
                                    transform=img_preprocess['train'],
                                    split='train')
    
    return train_set

def create_model_and_optimizer(model_type,device):
    print("model_type: ",model_type)
    model = instruct_model.Blip2T5InstructEnhance.load_pretrained_model(model_type=model_type)
    model.to(device)
    total_params = sum([param.nelement() for param in model.parameters()])
    print('Total Params: {:.3f}M'.format(total_params / 1000000))


    print('======= MODULE PRINT ========')
    for n, c in model.named_children():
            # print(n)
        print("[children module: {}]".format(n))
        for cn, p in c.named_parameters():
            print("  >>  " + n + '.' + cn, "|", p.size(), "|", p.requires_grad, "|", p.device)
        # print(user_module)
    print('=============================')

    num_gpus = torch.cuda.device_count()
        
    print("num_gpus: ",num_gpus)

    print("args.gpu: ",args.gpu)

    if args.world_size > 1:
       if args.gpu == 0:
           print('use {} gpus!'.format(args.gpu))
       
       print("DDP")
       model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer




def train(model, optimizer, dataloader, scaler, epoch, cur_step, scheduler,writer,device):
    global global_step
    model.train()
    avg_loss = utils.RunningAverage()
    #with tqdm(total=len(dataloader), disable=True) as t:
    with tqdm(total=len(dataloader), disable=False if args.rank == 0 else True) as t:
    #with tqdm(total=len(dataloader)) as t:
        dataloader.sampler.set_epoch(epoch)
        for step, data in enumerate(dataloader):
            global_step=global_step+1

            input_data,retrieval = data 
            
            image_data = input_data['image'].to(device)
            input_data ['image'] = image_data

            image_pos_data = retrieval['image_pos_data'].to(device)
            retrieval ['image_pos_data'] = image_pos_data

            image_neg_data = retrieval['image_neg_data'].to(device)
            retrieval ['image_neg_data'] = image_neg_data

            optimizer.zero_grad()
            with autocast():
                loss = model(input_data,retrieval)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step()

            if cur_step < args.warm_up_steps:
                scheduler.step(0, cur_step)
            else:
                skip_epoch = int(args.warm_up_steps // len(dataloader))
                scheduler.step(epoch-skip_epoch, cur_step)
            cur_step += 1
            
            avg_loss.update(loss.item())
            # #t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
            # print([global_step, optimizer.param_groups[0]['lr']])
            if args.rank == 0:
                writer.add_scalar("loss",loss.item(),global_step)
            if args.rank == 0 and global_step % 50 == 0:
                #print('global_step={}, loss={:05.3f}, lr={:05.10f}'.format(global_step, avg_loss(), optimizer.param_groups[0]['lr'])) 
                logging.info('global_step={}, loss={:05.3f}, lr={:05.10f}'.format(global_step, avg_loss(), optimizer.param_groups[0]['lr']))
            t.set_postfix(step='{}, loss={:05.10f}, lr={:05.10f}'.format(global_step, avg_loss(), optimizer.param_groups[0]['lr']))
            t.update()

            

    return avg_loss(),cur_step

def train_and_eval(args,model, optimizer, trainset,device):

    train_sampler = DistributedSampler(trainset)

    trainloader = dataloader.DataLoader(trainset, 
                                        sampler=train_sampler, 
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers, 
                                        pin_memory=False)
    
    scaler = GradScaler()
    epoches = args.num_epochs
    
    if args.rank == 0:    
        print("len(trainset) ", len(trainset))

        print("args.batch_size: ", args.batch_size)

        print('len(trainloader): ', len(trainloader))
    
    
    
    if args.rank == 0:
        writer=SummaryWriter(os.path.join(args.model_dir,"run"))
    else:
        writer = None

    cur_step = 0
    
    scheduler = LinearWarmupCosineLRScheduler(optimizer, max_epoch=args.num_epochs, min_lr=args.min_lr, init_lr=args.lr, warmup_steps=args.warm_up_steps, warmup_start_lr=args.init_lr)
    for epoch in range(epoches):
        
        if args.rank == 0:
            print("Epoch {}/{}".format(epoch + 1, epoches))
            logging.info("Epoch {}/{}".format(epoch + 1, epoches))

        the_loss,cur_step = train(model, optimizer, trainloader, scaler, epoch,cur_step, scheduler, writer, device)
        
        print("AVERAGE_LOSS={:05.3f}".format(the_loss)) 

        if args.rank == 0:
            logging.info("AVERAGE_LOSS={:05.3f}".format(the_loss))
            writer.add_scalar("AVERAGE_LOSS",the_loss,epoch)

        if args.rank == 0:
            print("epoch={}".format(epoch)) 
            print('==> Saving...')
            logging.info("epoch={}".format(epoch))
            if args.world_size==1:
                save_file = os.path.join(args.model_dir, 't5_model_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(model.state_dict(), save_file)
            else:
                save_file = os.path.join(args.model_dir, 't5_model_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(model.module.state_dict(), save_file)


if __name__ == '__main__':
    
    
    init_dis_mode(args)

    device = torch.device(args.device)
    print("device: ", device)
    print(args)
    print("args.rank: ",args.rank)
    print("args.local_rank: ",args.local_rank)

    args.model_dir= args.model_dir +'/'+ "fintune_"+str(args.lr)+"x"+str(args.batch_size)+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    if args.rank == 0 and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir,exist_ok=True)
    
    if args.rank == 0:
        print(args.model_dir)
        with open(os.path.join(args.model_dir, 'params.md'), 'w+') as f:
            f.write('params: \n```python \n')
            for k in args.__dict__.keys():
                p = "'{}'={}\n".format(k, args.__dict__[k])
                f.write(p)
            f.write('```')

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    if args.rank == 0:
        utils.set_logger(os.path.join(args.model_dir, 'train.log'))
        logging.info('Loading the datasets and model...')
        print('Loading the datasets and model...')
        logging.info("model_dir={}".format(args.model_dir))

    train_set = get_dataset(args)
    model, optimizer = create_model_and_optimizer(model_type=args.model_type,device=device)
    train_and_eval(args,model, optimizer, train_set, device)
