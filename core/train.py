import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import sys, copy, cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *
from torch.optim import SGD, Adam, lr_scheduler, AdamW
from datasets.metrics import MAE, IoU, ACC
from tensorboardX import SummaryWriter
import logging
from utils import seed_torch, denormalize, convert_bn_to_syncbn, setup_process_groups, LinearCosineAnnealingLR
from utils import tensor_pose_processing_edge, tensor_pose_processing_mask, tensor_pose_processing_gt, tensor_pose_processing_inpainting
from utils import save_weight, save_checkpoint
from torch import optim
import tqdm
from datetime import datetime, timedelta
from torchvision.utils import make_grid
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.dataset import get_dataset
from torch.cuda.amp import autocast, GradScaler
from models.losses import *
import json 

def build_model(opt):
    params = opt['model']['params']
    model = globals()[opt['model']['name']](**params)
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

def get_metric(metric_name):
    return globals()[metric_name]()

def build_dataloader(opt, dataset_key, world_size=None, rank=None):
    dataset_config = opt['dataset'][dataset_key]
    dataset = get_dataset(opt, dataset_key, rank)
    if dataset_config['istraining'] and world_size is not None and rank is not None:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=dataset_config['batch_size'], 
            num_workers=dataset_config['num_workers'], 
            sampler=sampler,
            drop_last=False
        )
        return dataloader, sampler
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=dataset_config['batch_size'], 
            num_workers=dataset_config['num_workers'], 
            shuffle=dataset_config['shuffle']
        )
        return dataloader, None

def aggregate_loss_records(loss_records, world_size):
    aggregated_loss_records = {}
    for key, values in loss_records.items():
        values_tensor = torch.tensor(values, dtype=torch.float32, device='cuda')
        dist.all_reduce(values_tensor, op=dist.ReduceOp.SUM)
        aggregated_loss_records[key] = (values_tensor / world_size).cpu().tolist()
    return aggregated_loss_records

def merge_data_batches(data_batches):
    merged_data = {}
    for key in data_batches[0].keys():
        if isinstance(data_batches[0][key], torch.Tensor):
            merged_data[key] = torch.cat([data_batch[key] for data_batch in data_batches], dim=0)
        else:
            sums = []
            for data_batch in data_batches:
                sums += data_batch[key]
            merged_data[key] = sums
    return merged_data

def shuffle_data(data):
    batch_size = data['image'].size()[0]
    indices = torch.randperm(batch_size)
    shuffled_data = {key: value[indices] for key, value in data.items()}
    return shuffled_data

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))

def iou_cal(pred, gt):
    pred = F.sigmoid(pred)
    intersection = torch.sum(pred * gt, dim=(1, 2, 3))
    union = torch.sum(pred + gt, dim=(1, 2, 3))
    iou = intersection / (union - intersection + 1e-6)
    return iou

def loss_computation(logits_list, data, losses, opt):
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
                target = data['gt']
            elif gt_input == "edge":
                target = data['edge']
            if loss_fn.__class__.__name__ == 'UALoss':
                loss = coef * loss_fn(logits, target, iter_percentage)
            elif loss_fn.__class__.__name__ == 'NCLoss':
                q = 1 if data['tmp_epoch'] > opt['training']['q_epoch'] else 2
                loss = coef * loss_fn(logits, target, q)
            elif loss_fn.__class__.__name__ == 'L1Loss' or loss_fn.__class__.__name__ == 'POTLoss':
                loss = coef * loss_fn(logits, target)
            else:
                loss = coef * loss_fn(logits, target)
            loss_list.append(loss)
    return loss_list

def train(rank, world_size, opt, loss_records):
    seed_torch(opt['seed'] + rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = build_model(opt)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=opt['training']['find_unused_parameters'])

    optimizer = build_optimizer(opt, model)
    scheduler = build_scheduler(opt, optimizer)

    dataloaders = {key: build_dataloader(opt, key, world_size, rank) for key in opt['dataset'] if 'test' != key}
    train_loaders = {key: dataloaders[key][0] for key in dataloaders.keys() if 'train' in key}
    print(len(train_loaders))
    samplers = {key: dataloaders[key][1] for key in dataloaders.keys() if 'train' in key}
    val_loader = dataloaders['val'][0]
    val_metrics = {metric_name: get_metric(metric_name) for metric_name in opt['dataset']['val']['metric']}

    start_epoch = 1
    if opt['training']['load'] is not None and os.path.exists(opt['training']['load']):
        checkpoint = torch.load(opt['training']['load'], map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Loaded checkpoint from {opt["training"]["load"]}, resuming training from epoch {start_epoch}')

    if opt['training']['pretrain'] is not None and os.path.exists(opt['training']['pretrain']):
        pretrin_weight = torch.load(opt['training']['pretrain'], map_location=device)
        model.load_state_dict(pretrin_weight['model'])
        print(f'Loaded pretrain model from {opt["training"]["pretrain"]}')

    loss_fns = []
    coefs = []
    gt_inputs = []

    for loss in opt['training']['losses'].values():
        if isinstance(loss['type'], list):
            loss_fns.append([globals()[lt]() for lt in loss['type']])
            coefs.append(loss['coef'])
            gt_inputs.append(loss['gt_input'])
    losses = {'types': loss_fns, 'coef': coefs, 'gt_input': gt_inputs}
    if rank == 0:
        logging.info("losses: %s" % losses)

    total_step = min([len(loader) for loader in train_loaders.values()])
    total_iters = total_step * opt['training']['epochs']
    finished_iters = 0
    save_path = opt['training']['save_path']
    if rank == 0:
        print("starting logging!")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(save_path, 'log.log'), mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 设置StreamHandler并将其日志级别设置为WARNING以避免在终端打印INFO日志
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        logging.info("Train")
        logging.info("Config %s" % opt)

    step = 0
    if rank == 0:
        writer = SummaryWriter(save_path + 'summary')

    scaler = GradScaler()

    for epoch in range(start_epoch, opt['training']['epochs'] + 1):
        for sampler in samplers.values():
            if sampler is not None:
                sampler.set_epoch(epoch)

        model.train()
        loss_all = 0
        epoch_step = 0
        data_end_time = datetime.now()
        lr = optimizer.param_groups[0]['lr']

        for i, data_batch in enumerate(zip(*train_loaders.values()), start=1):
            optimizer.zero_grad()
            iter_start_time = datetime.now()
            
            data = merge_data_batches(data_batch)
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            # data = shuffle_data(data)
            data['total_step'] = total_step
            data['tmp_iter'] = i
            data['tmp_epoch'] = epoch

            # with autocast():
            logits_list = model(data)['res']
            loss_list = loss_computation(logits_list, data, losses, opt)
            loss = sum(loss_list)
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            step += 1
            epoch_step += 1
            finished_iters = epoch_step + epoch * total_step
            loss_all += loss.item()

            # 写一个1%概率保存图像可视化观察结果
            if random.random() < 0.1:
                dest_img = data['image'][0].clone().cpu().data
                dest_img = (dest_img * 255.).to(torch.uint8)
                dest_img = dest_img.permute(1, 2, 0).numpy()
                cv2.imwrite(f'{save_path}/image.png', dest_img[:, :, ::-1])
                dest_gt = data['gt'][0].clone().cpu().data
                dest_gt = (dest_gt * 255.).to(torch.uint8)
                dest_gt = dest_gt.permute(1, 2, 0).numpy()
                cv2.imwrite(f'{save_path}/gt.png', dest_gt[:, :, ::-1])
                dest_pre = (logits_list[opt['training']['main_output_index']]).clone()[0].cpu().data
                dest_pre = (dest_pre * 255.).to(torch.uint8)
                dest_pre = dest_pre.permute(1, 2, 0).numpy()
                cv2.imwrite(f'{save_path}/pre.png', dest_pre[:, :, ::-1])

            if rank == 0 and (i % 20 == 0 or i == total_step or i == 1):
                
                iter_end_time = datetime.now()
                data_time = (iter_start_time - data_end_time).total_seconds()
                iter_time = (iter_end_time - iter_start_time).total_seconds()
                
                eta_seconds = (total_iters - finished_iters) * iter_time
                eta = str(timedelta(seconds=int(eta_seconds)))

                # print(f'{datetime.now()} | Epoch [{epoch:03d}/{opt["training"]["epochs"]:03d}], '
                #       f'Step [{i:04d}/{total_step:04d}], LR {lr:.8f}, Loss: {loss.item():.4f}, '
                #       f'DataTime: {data_time:.4f} sec, IterTime: {iter_time:.4f}sec, ImageShape: {data["image"].shape},  ETA: {eta}')
                logging.info(f'{datetime.now()} | Epoch [{epoch:03d}/{opt["training"]["epochs"]:03d}], '
                      f'Step [{i:04d}/{total_step:04d}], LR {lr:.8f}, Loss: {loss.item():.4f}, '
                      f'DataTime: {data_time:.4f} sec, IterTime: {iter_time:.4f}sec, ImageShape: {data["image"].shape},  ETA: {eta}')
                writer.add_scalar('Loss', loss.item(), global_step=step)

                # grid_image = make_grid(data['image'][0].clone().cpu().data, 1, normalize=True)
                grid_image = make_grid(denormalize(data['image'][0].clone().cpu().data), 1, normalize=True)
                writer.add_image('RGB', grid_image, step)

                grid_image = make_grid(data['gt'][0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)
            
                pre = (logits_list[opt['training']['main_output_index']]).sigmoid().clone()[0].cpu().data 
                pre = (pre * 255.).to(torch.uint8)
                grid_image_2 = make_grid(pre, 1, normalize=False)
                writer.add_image('Pre', grid_image_2, step)

                grid_image = make_grid(denormalize(data['image'][1].clone().cpu().sigmoid().data), 1, normalize=True)
                writer.add_image('RGB2', grid_image, step)
            data_end_time = datetime.now()
        scheduler.step()

        loss_all /= epoch_step
        if rank == 0:
            logging.info(f'[Train Info]: Epoch [{epoch:03d}/{opt["training"]["epochs"]:03d}], Loss_AVG: {loss_all:.4f}')
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, save_path, opt)
        if rank == 0 and (epoch % opt['training']['val_step'] == 0):
            val(val_loader, model, epoch, save_path, writer, opt, val_metrics)


    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def val(test_loader, model, epoch, save_path, writer, opt, metrics):
    torch.cuda.empty_cache()
    model.eval()
    process_func = globals()[opt['dataset']['val']['pose_process'][0]]
    gt_process = tensor_pose_processing_gt

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(test_loader, start=1)):
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            outputs = model(data)
            res = outputs['res'][opt['training']['main_output_index']]
            
            for j in range(len(res)):
                pre_single = process_func(res[j], data, j)
                gt_single = gt_process(data['gt'][j], data, j)
                
                for metric in metrics.values():
                    metric.update(pre_single, gt_single)
        
        # 动态计算并记录每个指标，打印的时候一行打印完
        for metric_name, metric in metrics.items():
            final_value = metric.compute()
            writer.add_scalar(f'Validation/{metric_name}', final_value, global_step=epoch)

            print(f'Epoch: {epoch}, {metric_name}: {final_value}, Best {metric_name}: {metric.best_score} at epoch {metric.best_epoch}')
            logging.info(f'Epoch: {epoch}, {metric_name}: {final_value}, Best {metric_name}: {metric.best_score} at epoch {metric.best_epoch}')
        
        # 比较当前MAE和最优MAE并保存最佳模型
        # if 'MAE' in metrics and metrics['MAE'].compute() < metrics['MAE'].best_score:
        #     metrics['MAE'].best_score = metrics['MAE'].compute()
        #     metrics['MAE'].best_epoch = epoch
        #     weight = {'model': model.state_dict()}
        #     torch.save(weight, os.path.join(save_path, 'model_best.pth'))
        #     print(f'Save state_dict successfully! Best epoch: {epoch}.')

        for metric_name, metric in metrics.items():
            metric.reset()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        opt = yaml.safe_load(file)

    seed_torch(opt['seed'])

    if not os.path.exists(opt['training']['save_path']):
        os.makedirs(opt['training']['save_path'])

    world_size = len(opt['training']['gpu_id'].split(','))
    os.environ['MASTER_ADDR'] = str(opt['training']['MASTER_ADDR'])
    os.environ['MASTER_PORT'] = str(opt['training']['MASTER_PORT'])
    gpu_ids = opt['training']['gpu_id'].split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    loss_records = {}
    mp.spawn(train, args=(world_size, opt, loss_records), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
