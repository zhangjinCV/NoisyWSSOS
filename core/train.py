import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *
from torch.optim import *
from tensorboardX import SummaryWriter
import logging
from utils import seed_torch, denormalize, convert_bn_to_syncbn, setup_process_groups, LinearCosineAnnealingLR
from utils import save_weight, save_checkpoint
from torch import optim
import tqdm
from torchvision.utils import make_grid
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.dataset import get_dataset, seed_torch
from torch.cuda.amp import autocast, GradScaler
from models.losses import *
import json 


def build_model(opt):
    model = globals()[opt['model']['name']]()
    return model

def build_optimizer(opt, model):
    optimizer_class = getattr(optim, opt['optimizer']['type'])
    optimizer = optimizer_class(model.parameters(), **opt['optimizer']['params'])
    return optimizer

def build_scheduler(opt, optimizer):
    if opt['scheduler']['type'] == 'LinearCosineAnnealingLR':
        scheduler_class = LinearCosineAnnealingLR
    else:
        scheduler_class = getattr(optim.lr_scheduler, opt['scheduler']['type'])
    
    scheduler = scheduler_class(optimizer, **opt['scheduler']['params'])
    return scheduler

def build_dataloader(opt, dataset_type, world_size=None, rank=None):
    datasets = []
    for key in opt['dataset']:
        if key.startswith(dataset_type):
            datasets.append(get_dataset(opt, key))
    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    if dataset_type.startswith('train') and world_size is not None and rank is not None:
        sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank)
        dataloader = torch.utils.data.DataLoader(
            dataset=combined_dataset, 
            batch_size=opt['dataloader']['batch_size'], 
            num_workers=opt['dataloader']['num_workers'], 
            sampler=sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=combined_dataset, 
            batch_size=opt['dataloader']['batch_size_val'], 
            num_workers=opt['dataloader']['num_workers'], 
            shuffle=opt['dataloader']['shuffle']
        )

    return dataloader

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))

def loss_computation(logits_list, targets, edges, losses, data):
    check_logits_losses(logits_list, losses)
    loss_list = []
    iter_percentage = data['tmp_iter'] / data['total_step']
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_fns = losses['types'][i]
        coefs = losses['coef'][i]
        gt_inputs = losses['gt_input'][i]
        for loss_fn, coef, gt_input in zip(loss_fns, coefs, gt_inputs):
            if gt_input == "gt":
                target = targets
            elif gt_input == "edge":
                target = edges
            if loss_fn.__class__.__name__ == 'UALoss':
                loss = coef * loss_fn(logits, target, iter_percentage)
            else:
                loss = coef * loss_fn(logits, target)
            loss_list.append(loss)
    return loss_list

def train(rank, world_size, opt):
    seed_torch(2024 + rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = build_model(opt)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = build_optimizer(opt, model)
    scheduler = build_scheduler(opt, optimizer)
    train_loader = build_dataloader(opt, 'train', world_size, rank)
    val_loader = build_dataloader(opt, 'val')

    start_epoch = 1
    if opt['training']['load'] is not None and os.path.exists(opt['training']['load']):
        checkpoint = torch.load(opt['training']['load'], map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint from {opt["training"]["load"]}, resuming training from epoch {start_epoch}')

    loss_fns = []
    coefs = []
    gt_inputs = []

    for loss in opt['training']['losses'].values():
        if isinstance(loss['type'], list):
            loss_fns.append([globals()[lt]() for lt in loss['type']])
            coefs.append(loss['coef'])
            gt_inputs.append(loss['gt_input'])
    
    losses = {'types': loss_fns, 'coef': coefs, 'gt_input': gt_inputs}
    logging.info("losses: %s" % losses)

    total_step = len(train_loader)
    save_path = opt['training']['save_path']
    if rank == 0:
        print("starting logging!")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(save_path, 'log.log'), mode='a')
        formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info("Train")
        logging.info("Config %s" % opt)

    step = 0
    if rank == 0:
        writer = SummaryWriter(save_path + 'summary')

    scaler = GradScaler()

    for epoch in range(start_epoch, opt['training']['epochs'] + 1):
        train_loader.sampler.set_epoch(epoch)

        model.train()
        loss_all = 0
        epoch_step = 0
        lr = optimizer.param_groups[0]['lr']

        for i, data in enumerate(train_loader, start=1):
            data = {k: v.cuda(device=device) for k, v in data.items()}
            data['total_step'] = total_step
            data['tmp_iter'] = i
            with autocast():
                logits_list = model(data)['res']
                loss_list = loss_computation(logits_list, data['gt'], data['edge'], losses, data)
                loss = sum(loss_list)
            scaler.scale(loss).backward()
            
            if i % opt['training']['grad_accum'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if rank == 0 and (i % 20 == 0 or i == total_step or i == 1):
                print(f'{datetime.now()} Epoch [{epoch:03d}/{opt["training"]["epochs"]:03d}], Step [{i:04d}/{total_step:04d}], LR {lr:.8f} Loss: {loss.item():.4f}')
                logging.info(f'[Train Info]:Epoch [{epoch}/{opt["training"]["epochs"]}], Step [{i}/{total_step}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss', loss.item(), global_step=step)

                grid_image = make_grid(denormalize(data['image'][0].clone().cpu().data), 1, normalize=True)
                writer.add_image('RGB', grid_image, step)

                grid_image = make_grid(data['gt'][0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                pre = (logits_list[4]).clone().sigmoid().cpu().data * 255.
                pre = pre.to(torch.uint8)[0]
                grid_image_2 = make_grid(pre, 1, normalize=False)
                writer.add_image('Pre', grid_image_2, step)

        scheduler.step()

        loss_all /= epoch_step
        if rank == 0:
            logging.info(f'[Train Info]: Epoch [{epoch:03d}/{opt["training"]["epochs"]:03d}], Loss_AVG: {loss_all:.4f}')
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        save_checkpoint(model, optimizer, scheduler, epoch, save_path, opt)

        if rank == 0 and (epoch % opt['training']['val_step'] == 0):
            val(val_loader, model, epoch, save_path, writer, opt)

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


def val(test_loader, model, epoch, save_path, writer, opt):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        mae_sum = []
        for i, data in tqdm.tqdm(enumerate(test_loader, start=1)):
            data = {k: v.cuda() for k, v in data.items()}
            outputs = model(data)
            res = outputs['res'][4]
            res = res.sigmoid()
            for j in range(len(res)):
                pre = F.interpolate(res[j].unsqueeze(0), size=(data['H'][j].item(), data['W'][j].item()), mode='bilinear')
                gt_single = F.interpolate(data['gt'][j].unsqueeze(0), size=(data['H'][j].item(), data['W'][j].item()), mode='bilinear')
                mae_sum.append(torch.mean(torch.abs(gt_single - pre)).item())

        mae = np.mean(mae_sum)
        mae = "%.5f" % mae
        mae = float(mae)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print(f'Epoch: {epoch}, MAE: {mae}, bestMAE: {opt["training"]["best_mae"]}, bestEpoch: {opt["training"]["best_epoch"]}.')
        if mae < opt['training']['best_mae']:
            opt['training']['best_mae'] = mae
            opt['training']['best_epoch'] = epoch
            torch.save(model.state_dict(), save_path + 'model_best.pth')
            print(f'Save state_dict successfully! Best epoch: {epoch}.')
        logging.info(f'[Val Info]:Epoch: {epoch}, MAE: {mae}, bestMAE: {opt["training"]["best_mae"]}, bestEpoch: {opt["training"]["best_epoch"]}.')
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        opt = yaml.safe_load(file)

    assert opt['training']['grad_accum'] >= 1
    seed_torch(opt['seed'])

    if not os.path.exists(opt['training']['save_path']):
        os.makedirs(opt['training']['save_path'])

    world_size = len(opt['training']['gpu_id'].split(','))
    os.environ['MASTER_ADDR'] = str(opt['training']['MASTER_ADDR'])
    os.environ['MASTER_PORT'] = str(opt['training']['MASTER_PORT'])
    gpu_ids = opt['training']['gpu_id'].split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    mp.spawn(train, args=(world_size, opt), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
