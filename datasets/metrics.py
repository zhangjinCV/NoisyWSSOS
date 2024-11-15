import numpy as np
import os 
import cv2 


class MAE:
    def __init__(self):
        self.reset()
        self.best_score = 1.0
        self.best_epoch = 0

    def reset(self):
        self.maes = []
        
    def update(self, pred, gt):
        pred = pred / 255.
        gt = gt / 255.
        mae = np.abs(pred - gt).mean()
        self.maes.append(mae)

    def compute(self):
        return np.mean(self.maes)


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
    

class ACC:
    def __init__(self):
        self.reset()    
        self.best_score = 0.0
        self.best_epoch = 0

    def reset(self):
        self.accs = []

    def update(self, pred, gt):
        pred = pred / 255.
        gt = gt / 255.
        gt = np.where(gt > 0.5, 1, 0)
        pred = np.where(pred > 0.5, 1, 0)
        acc = (pred == gt).mean()
        self.accs.append(acc)

    def compute(self):
        return np.mean(self.accs)