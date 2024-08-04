import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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
from utils import seed_torch, denormalize, tensor_pose_processing_mask, tensor_pose_processing_edge, tensor_pose_processing_inpainting
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import logging
import tqdm
from torch.cuda.amp import autocast
from datasets.dataset import get_dataset

def build_model(opt):
    params = opt['model']['params']
    model = globals()[opt['model']['name']](**params)
    return model

def build_dataloader(opt, dataset_key):
    dataset_config = opt['dataset'][dataset_key]
    
    if isinstance(dataset_config['image_root'], list):
        datasets = [get_dataset({**dataset_config, 'image_root': img_root, 'gt_root': gt_root, 'file_list': file_list}) 
                    for img_root, gt_root, file_list in zip(dataset_config['image_root'], dataset_config['gt_root'], dataset_config['file_list'])]
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        dataset = get_dataset(dataset_config)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=dataset_config['batch_size'],
        num_workers=dataset_config['num_workers'],
        shuffle=dataset_config['shuffle']
    )
    return dataloader

def load_pretrained_weight(model, opt):
    checkpoint = torch.load(opt, map_location='cpu')['model']
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print(f'Loaded pretrained weight from {opt}')
    return model

def val(opt):
    seed_torch(opt['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(opt)
    model.to(device)

    torch.cuda.empty_cache()
    load_pretrained_weight(model, opt['dataset']['test']['load'])

    model.eval()

    with torch.no_grad():
        for key in opt['dataset']:
            if 'test' in key:
                for img_root, gt_root, file_list, save_path in zip(opt['dataset']['test']['image_root'], opt['dataset']['test']['gt_root'], opt['dataset']['test']['file_list'], opt['dataset']['test']['save_path']):
                    print(f"Testing on {img_root} dataset")
                    opt['dataset'][key]['image_root'] = img_root
                    opt['dataset'][key]['gt_root'] = gt_root
                    opt['dataset'][key]['file_list'] = file_list
                    val_loader = build_dataloader(opt, key)
                    
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
