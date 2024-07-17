import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *
from utils import seed_torch, denormalize, tensor_pose_processing_mask, tensor_pose_processing_edge
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import logging
import tqdm
from torch.cuda.amp import autocast
from datasets.dataset import get_dataset

def build_model(opt):
    model = globals()[opt['model']['name']]()
    return model

def build_dataloader(opt, dataset_key):
    dataset = get_dataset(opt, dataset_key)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt['dataloader']['batch_size_val'],
        num_workers=opt['dataloader']['num_workers'],
        shuffle=opt['dataloader']['shuffle']
    )
    return dataloader

def val(opt):
    seed_torch(opt['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(opt)
    model.to(device)

    if opt['dataset']['test']['load'] is not None and os.path.exists(opt['dataset']['test']['load']):
        checkpoint = torch.load(opt['dataset']['test']['load'], map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(f'Loaded checkpoint from {opt['dataset']['test']['load']}')

    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():
        for key in opt['dataset']:
            if 'test' in key:
                val_loader = build_dataloader(opt, 'test')
                save_path = opt['dataset']['test']['save_path']

                save_indices = opt['dataset']['test']['save_indices']
                save_index_names = opt['dataset']['test']['save_index_names']
                pose_process = [globals()[func_name] for func_name in opt['dataset']['test']['pose_process']]

                save_paths = [os.path.join(save_path, name) for name in save_index_names]
                for path in save_paths:
                    if not os.path.exists(path):
                        os.makedirs(path)

                for i, data in tqdm.tqdm(enumerate(val_loader, start=1)):
                    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    outputs = model(data)['res']
                    for idx, name, path, process_func in zip(save_indices, save_index_names, save_paths, pose_process):
                        res = outputs[idx]
                        for j in range(len(res)):
                            pre_img = process_func(res[j], data, j)
                            save_file_path = os.path.join(path, data["name"][j].replace('.jpg', '.png'))
                            cv2.imwrite(save_file_path, pre_img)

def main():
    parser = argparse.ArgumentParser(description='Testing configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        opt = yaml.safe_load(file)

    seed_torch(opt['seed'])

    val(opt)

if __name__ == '__main__':
    main()
