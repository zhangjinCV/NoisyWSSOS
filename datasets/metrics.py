import numpy as np
import os 
import cv2 


class MAE:
    def __init__(self):
        self.reset()
        self.best_score = 1.0
        self.best_epoch = 0

    def reset(self):
        self.sum = 0
        self.count = 0
        
    def update(self, pred, gt):
        pred = pred / 255.
        gt = gt / 255.
        self.sum += np.abs(pred - gt).mean()
        self.count += 1

    def compute(self):
        return self.sum / self.count


class IoU:
    def __init__(self):
        self.reset()
        self.best_score = 0.0
        self.best_epoch = 0

    def reset(self):
        self.ious = []

    def update(self, pred, gt):
        pred = pred / 255.
        gt = gt / 255.
        inter = (pred * gt).sum()
        union = (pred + gt).sum() - inter
        self.ious.append(inter / union)

    def compute(self):
        return np.mean(self.ious)
    