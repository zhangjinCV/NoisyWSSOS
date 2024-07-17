import math
import torch
import numpy as np
from thop import profile
from thop import clever_format
import random 
import os 
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


class LinearCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, warmup_epochs, total_epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(LinearCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        
        if current_epoch < self.warmup_epochs:
            # Linear warmup phase
            warmup_lr = [
                self.min_lr + (self.max_lr - self.min_lr) * current_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine annealing phase
            cosine_epoch = current_epoch - self.warmup_epochs
            cosine_total_epochs = self.total_epochs - self.warmup_epochs
            cosine_lr = [
                self.min_lr + 0.5 * (self.max_lr - self.min_lr) * 
                (1 + math.cos(math.pi * cosine_epoch / cosine_total_epochs))
                for base_lr in self.base_lrs
            ]
            return cosine_lr


def tensor_pose_processing_mask(tensor, data, j):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.sigmoid()
    tensor = F.interpolate(tensor, size=(data['W'][j].item(), data['H'][j].item()), mode='bilinear')
    tensor = tensor.cpu().numpy().squeeze() * 255
    np_one = tensor.astype(np.uint8)
    return np_one

def tensor_pose_processing_edge(tensor, data, j):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(data['W'][j].item(), data['H'][j].item()), mode='bilinear')
    tensor = tensor.cpu().numpy().squeeze() * 255
    np_one = tensor.astype(np.uint8)
    return np_one

def save_checkpoint(model, optimizer, scheduler, epoch, path, opt):
    save_iter = opt['save_iter']
    if epoch % save_iter == 0:
        save_path = os.path.join(path, 'epoch_{}.pth'.format(epoch))
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print('Checkpoint saved at {}'.format(save_path))
    
    checkpoints = [f for f in os.listdir(path) if f.endswith('.pth')]
    checkpoints = [f for f in checkpoints if 'epoch' in f]
    nums_save_pth = len(checkpoints)
    max_save_num = opt['max_save_num']
    if nums_save_pth > max_save_num:
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        for checkpoint in checkpoints[:nums_save_pth - max_save_num]:
            os.remove(os.path.join(path, checkpoint))
            print('Checkpoint removed at {}'.format(os.path.join(path, checkpoint)))

def save_weight(model, epoch, path, opt):
    save_iter = opt['save_iter']
    if epoch % save_iter == 0:
        save_path = path + '/epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), save_path)
        print('Model saved at {}'.format(save_path))
    
    pths = os.listdir(path)
    pths = [i for i in pths if '.pth' in i]
    nums_save_pth = [i for i in pths if 'epoch' in i]
    max_save_num = opt['max_save_num']

    if nums_save_pth > max_save_num:
        pths = os.listdir(path)
        pths = [i for i in pths if '.pth' in i]
        pths = [i for i in pths if 'epoch' in i]
        pths = [int(pth.split('_')[1].split('.')[0]) for pth in pths]
        pths.sort()
        for pth in pths[:nums_save_pth-max_save_num]:
            os.remove(path + '/epoch_{}.pth'.format(pth))
            print('Model removed at {}'.format(path + '/epoch_{}.pth'.format(pth)))



def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def convert_bn_to_syncbn(model, process_group):
    # Convert Batch Normalization layers to SyncBatchNorm layers
    return nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

def setup_process_groups():
    # Assuming you have 8 GPUs and want to split them into two groups
    ranks = list(range(4))
    r1, r2 = ranks[:2], ranks[2:]
    process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
    return process_groups


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = decay*init_lr
#         lr=param_group['lr']
#     return lr


def adjust_lr(now_epoch, top_epoch, max_epoch, init_lr, top_lr, min_lr, optimizer):
    mum_step = top_epoch
    min_lr = min_lr
    max_lr = top_lr
    total_steps = max_epoch
    if now_epoch < mum_step:
        lr = min_lr + abs(max_lr - min_lr) / (mum_step) * now_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            lr=param_group['lr']
        return lr
    else:
        progress = (now_epoch - mum_step) / (total_steps - mum_step)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))  
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            lr=param_group['lr']
        return lr
    

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

