import torch 
from torch import nn 
import numpy as np 
from torch.nn import functional as F



class UALoss(nn.Module):
    def __init__(self):
        super(UALoss, self).__init__()

    def get_coef(self, iter_percentage, method='cos'):
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

    def forward(self, seg_logits, seg_gts, iter_percentage):
        ual_coef = self.get_coef(iter_percentage)
        sigmoid_x = seg_logits.sigmoid()
        loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
        return loss_map.mean() * ual_coef


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class NCLoss(nn.Module):
    def __init__(self):
        super(NCLoss, self).__init__()

    def wbce_loss(self, preds, targets):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        wbce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        return wbce.mean()

    def forward(self, preds, targets, q):
        wbce = self.wbce_loss(preds, targets)
        preds = torch.sigmoid(preds)
        preds_flat = preds.contiguous().view(preds.shape[0], -1)
        targets_flat = targets.contiguous().view(targets.shape[0], -1)
        numerator = torch.sum(torch.abs(preds_flat - targets_flat) ** q, dim=1)
        intersection = torch.sum(preds_flat * targets_flat, dim=1)
        denominator = torch.sum(preds_flat, dim=1) + torch.sum(targets_flat, dim=1) - intersection + 1e-6
        loss = numerator / denominator
        if q == 2:
            return loss.mean() + wbce
        else:
            return loss.mean() * 2


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        smooth = 1
        p = 2
        valid_mask = torch.ones_like(target)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
        loss = 1 - num / den
        return loss.mean()

