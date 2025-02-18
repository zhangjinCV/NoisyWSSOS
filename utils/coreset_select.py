import os, cv2, tqdm, glob
import numpy as np
from scipy.stats import norm, entropy
import torch 
from torch import nn 
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import wasserstein_distance

def read_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    image = cv2.resize(image, (384, 384))
    mask = cv2.resize(mask, (384, 384))
    return image, mask

def extract_foreground(image, mask):
    # 提取前景（框内像素）
    foreground = image[mask == 255]  # 提取前景像素 (n_pixels, 3)
    return foreground

def extract_background(image, mask):
    # 提取背景（框外像素）
    background = image[mask == 0]  # 提取背景像素 (n_pixels, 3)
    return background

def calculate_emd(feature1, feature2):
    """
    计算Earth Mover's Distance (EMD)。
    将多维特征展平成单维分布进行处理。
    """
    feature1_flat = feature1.flatten()  # 展平为1D
    feature2_flat = feature2.flatten()  # 展平为1D

    # 计算EMD
    emd = wasserstein_distance(feature1_flat, feature2_flat)
    return emd

def split_foreground(foreground, num_splits=3):
    sub_foregrounds = np.array_split(foreground, num_splits, axis=0)
    return sub_foregrounds

def calculate_average_emd(sub_foregrounds):
    """
    计算前景子块两两之间的EMD均值。
    """
    emd_values = []
    n = len(sub_foregrounds)
    for i in range(n):
        for j in range(i + 1, n):
            emd = calculate_emd(sub_foregrounds[i], sub_foregrounds[j])
            emd_values.append(emd)
    return np.mean(emd_values) if emd_values else 0

def main(image_dir, mask_dir, save_path, num_splits=3):
    image_paths = sorted(image_dir)
    mask_paths = sorted(mask_dir)

    results = []

    for image_path, mask_path in tqdm.tqdm(zip(image_paths, mask_paths)):
        image, mask = read_image_and_mask(image_path, mask_path)

        # 提取前景和背景
        foreground = extract_foreground(image, mask)
        background = extract_background(image, mask)
        if foreground.size <100 or background.size == 0:
            results.append((os.path.basename(image_path), 0))
        else:
        # 计算前景与背景的EMD
            fg_bg_emd = calculate_emd(foreground, background)
            # 分割前景区域为 num_splits x num_splits 块
            sub_foregrounds = split_foreground(foreground, num_splits)
            # 计算前景子块的平均EMD
            fg_self_emd = calculate_average_emd(sub_foregrounds)
            results.append((os.path.basename(image_path), 0.4 * fg_bg_emd + 0.6 * fg_self_emd))
    results.sort(key=lambda x: x[1], reverse=True)  # EMD值越高表示差异越大

    with open(save_path, 'w') as f:
        for filename, score in results:
            f.write(f'{filename}: {score}\n')

def TOP_coreset(S_path, alpha, beta, k, path_img, save_path):
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)
    
    # Calculate the number of elements to select
    num = int(n * alpha)
    left = int(n * ((1 - alpha) / 2))

    sorted_S = sorted(S, key=lambda x: x[1], reverse=False)
    selected = sorted_S[:num]
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in selected:
            f.write(f"{s[0]}\n")



def Tail_coreset(S_path, alpha, beta, k, path_img, save_path):
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)
    
    # Calculate the number of elements to select
    num = int(n * alpha)
    left = int(n * ((1 - alpha) / 2))

    sorted_S = sorted(S, key=lambda x: x[1], reverse=True)
    selected = sorted_S[:num]
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in selected:
            f.write(f"{s[0]}\n")



def Random_coreset(S_path, alpha, beta, k, path_img, save_path):
    import random
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)
    
    # Calculate the number of elements to select
    num = int(n * alpha)
    left = int(n * ((1 - alpha) / 2))

    sorted_S = sorted(S, key=lambda x: x[1], reverse=True)
    selected = random.sample(sorted_S, num)
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in selected:
            f.write(f"{s[0]}\n")


def CCS_coreset(S_path, alpha, beta, k, path_img, save_path):
    import random
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)
    
    # Calculate the number of elements to select
    num = int(n * alpha)
    ration = [int(0.2 * num), int(0.4 * num), int(0.2 * num), int(0.2 * num)]
    left = int(n * ((1 - alpha) / 2))
    sorted_S = sorted(S, key=lambda x: x[1], reverse=False)
    selected = []
    i = 0
    for nu in ration:
        sos = sorted_S[int(len(sorted_S) * i): int(len(sorted_S) * (i + 0.25))]
        selected += random.sample(sos, nu)
        i += 0.25
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in selected:
            f.write(f"{s[0]}\n")


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

def MDE_coreset(S_path, alpha, beta, k, path_img, save_path):
    # Load dataset
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)

    # Calculate the number of elements to select
    nums = [0.2 * alpha, 0.4 * alpha, 0.2 * alpha, 0.2 * alpha]
    nums = [int(i * n) for i in nums]

    # Step 2: Prune hard examples
    sorted_S = sorted(S, key=lambda x: x[1], reverse=False)
    num_pruned = int(n * (1 - beta))
    S_prime = sorted_S[:num_pruned]

    # Split into k subsets
    R = np.array_split(S_prime, k)
    S_c = []

    # Select samples based on maximum distance
    for i in range(len(R)):
        selected = select_according_max_distance_torch([i[0] for i in R[i]], nums[i], path_img)
        S_c.extend(selected)

    # Save the selected samples to a file
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in S_c:
            f.write(f"{s}\n")

def extract_features_from_image_batch(image_paths, model, transform, device):
    """
    Extract features from a batch of images using a pre-trained model.
    """
    images = [cv2.imread(image_path) for image_path in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = [cv2.resize(img, (224, 224)) for img in images]  # Resize to 224x224
    images = [transform(img) for img in images]
    images = torch.stack(images).cuda()  # Stack all images into a single batch tensor

    # Get features from the model (excluding the classification head)
    with torch.no_grad():
        features = model(images)
    return features  # This will return a tensor of shape (batch_size, feature_dim)


def wasserstein_distance_torch(x, y):
    """
    Calculate the 1D Wasserstein distance between two distributions.
    """
    l2 = torch.nn.functional.l1_loss(x, y)
    return l2


def select_according_max_distance_torch(R, num, path_img, device='cuda:0'):
    """
    Select samples from R based on maximum distance in feature space.
    """
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model = model.cuda()
    model.eval()

    # Define image transformations for the input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Precompute features for all images in R
    image_paths = [path_img + img for img in R]
    all_features = extract_features_from_image_batch(image_paths, model, transform, device)

    # Initialize selection
    selected = [R[0]]  # Select the first sample
    selected_features = all_features[0:1]  # Corresponding feature
    remaining_features = all_features[1:]  # Remaining features
    R = R[1:]  # Remove first sample from R

    # Iteratively select samples until the desired number is reached
    while len(selected) < num:
        # Calculate distances to all selected features (pairwise distance)
        distances = []
        for feature in remaining_features:
            dist = torch.min(torch.stack([wasserstein_distance_torch(feature, selected_feat) for selected_feat in selected_features]))
            distances.append(dist)
        
        # Find the sample with the maximum distance
        max_idx = torch.argmax(torch.tensor(distances)).item()
        selected.append(R[max_idx]) 
        print(R[max_idx])
        
        # Add the selected sample's feature to the selected_features tensor
        selected_features = torch.cat([selected_features, remaining_features[max_idx].unsqueeze(0)])
        
        # Remove the selected sample from the remaining pool
        R = R[:max_idx] + R[max_idx + 1:]
        remaining_features = torch.cat([remaining_features[:max_idx], remaining_features[max_idx + 1:]])
        
        print(f"Selected {len(selected)} samples")
        
    return selected


def remaining_set(full, coreset, save_path):
    with open(full, 'r') as f:
        full = f.readlines()
    with open(coreset, 'r') as f:
        coreset = f.readlines()
    full = [s.strip() for s in full]
    coreset = [s.strip() for s in coreset]
    remaining = [f for f in full if f not in coreset]
    with open(save_path, 'w') as f:
        for r in remaining:
            f.write(f"{r}\n")


def generate_namelist(path, save_path):
    name = os.listdir(path)
    with open(save_path, 'w') as f:
        for r in name:
            f.write(f"{r}\n")


if __name__ == '__main__':
    images = '/mnt/jixie16t/dataset/COD/CAMO_COD_train/image/'
    boxes = '/mnt/jixie16t/dataset/COD/CAMO_COD_train/box/'
    CCS_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt", 0.2, 0, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_cod')
    remaining_set("/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_cod20.txt", '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/remaining80_cod.txt')
    

    images = '/mnt/jixie16t/dataset/DIS5K/DIS-TR/im/'
    CCS_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt", 0.2, 0, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_dis5k')

    remaining_set("/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_dis5k20.txt", '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/remaining80_dis5k.txt')

    images = '/mnt/jixie16t/dataset/Polyp/TrainDataset/images/'
    CCS_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.2, 0, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_polyp')
    
    remaining_set("/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/ccs_polyp20.txt", '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/AblationStudy/DifferentSelectionStrategy/ccs/remaining80_polyp.txt')

    # scores = main(glob.glob(os.path.join(images, '*.jpg')), glob.glob(os.path.join(boxes, '*.png')), '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt", 0.2, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt", 0.01, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset')
    # remaining_set('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset1.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining99.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt", 0.05, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset')
    # remaining_set('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset5.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining95.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset_select.txt", 0.1, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset')
    # # remaining_set('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset20.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining80.txt')
    # remaining_set('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset10.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining90.txt')

    # # DIS5K
    # images = '/mnt/jixie16t/dataset/DIS5K/DIS-TR/im/'
    # boxes = '/mnt/jixie16t/dataset/DIS5K/DIS-TR/box/'
    # scores = main(glob.glob(os.path.join(images, '*.jpg')), glob.glob(os.path.join(boxes, '*.png')), '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt", 0.2, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt", 0.01, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt", 0.05, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset_select.txt", 0.1, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset')
    # remaining_set('/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset20.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining80.txt')
    # remaining_set('/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset1.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining99.txt')
    # remaining_set('/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset5.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining95.txt')
    # remaining_set('/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset10.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining90.txt')

    # # PoyP
    # images = '/mnt/jixie16t/dataset/Polyp/TrainDataset/image/'
    # boxes = '/mnt/jixie16t/dataset/Polyp/TrainDataset/box/'
    # scores = main(glob.glob(os.path.join(images, '*.jpg')), glob.glob(os.path.join(boxes, '*.png')), '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.2, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.01, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.05, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.1, 0.2, 4, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset20.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining80.txt')
    # remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset1.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining99.txt')
    # remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset5.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining95.txt')
    # remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset10.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining90.txt')
    # generate_namelist("/mnt/jixie16t/dataset/Polyp/TestDataset/Kvasir/image", '/mnt/jixie16t/dataset/Polyp/TestDataset/Kvasir/namelist.txt')
    # generate_namelist("/mnt/jixie16t/dataset/Polyp/TestDataset/ETIS-LaribPolypDB/image", '/mnt/jixie16t/dataset/Polyp/TestDataset/ETIS-LaribPolypDB/namelist.txt')
    # generate_namelist("/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-ColonDB/image", '/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-ColonDB/namelist.txt')
    # generate_namelist("/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-300/image", '/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-300/namelist.txt')
    # generate_namelist("/mnt/jixie16t/dataset/COD/CHAMELEON_TestingDataset/image", "/mnt/jixie16t/dataset/COD/CHAMELEON_TestingDataset/namelist.txt")
    # generate_namelist("/mnt/jixie16t/dataset/DIS5K/DIS-TE1/im", "/mnt/jixie16t/dataset/DIS5K/DIS-TE1/namelist.txt")
    # generate_namelist("/mnt/jixie16t/dataset/DIS5K/DIS-TE2/im", "/mnt/jixie16t/dataset/DIS5K/DIS-TE2/namelist.txt")
    # generate_namelist("/mnt/jixie16t/dataset/DIS5K/DIS-TE3/im", "/mnt/jixie16t/dataset/DIS5K/DIS-TE3/namelist.txt")
    # generate_namelist("/mnt/jixie16t/dataset/DIS5K/DIS-TE4/im", "/mnt/jixie16t/dataset/DIS5K/DIS-TE4/namelist.txt")
