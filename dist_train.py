import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from core.model import WeTr
from datasets import voc
from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)



def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def validate(model=None, criterion=None, data_loader=None):

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs,
                                            size=labels.shape[1:],
                                            mode='bilinear',
                                            align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(
                torch.argmax(resized_outputs,
                             dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = eval_seg.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(cfg):

    num_workers = 4

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    '''
    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
    else:
        print('Using CPU:')
        device = torch.device('cpu')
    '''
    device  =torch.device(args.local_rank)
    wetr = WeTr(backbone=cfg.exp.backbone,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True)
    param_groups = wetr.get_param_groups()
    
    wetr.to(device)
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    #wetr, optimizer = amp.initialize(wetr, optimizer, opt_level="O1")
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    criterion = criterion.to(device)

    train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)

    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    for n_iter in range(cfg.train.max_iters):
        
        try:
            _, inputs, labels = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            _, inputs, labels = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = wetr(inputs)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = criterion(outputs, labels.type(torch.long))

        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0 and args.local_rank==0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(n_iter+1, delta, eta, lr, seg_loss.item()))
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            val_loss, val_score = validate(model=wetr, criterion=criterion, data_loader=val_loader)
            if args.local_rank==0:
                logging.info(val_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.local_rank == 0:
        setup_logger()
        logging.info('\nconfigs: %s' % cfg)
    #setup_seed(1)
    
    train(cfg=cfg)
