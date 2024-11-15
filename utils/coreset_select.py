import os, cv2, tqdm, glob
import numpy as np
from scipy.stats import norm, entropy
import torch 
from torch import nn 
from torch.nn import functional as F


def get_sim_socre_according_foreground_background(images, boxes, save_path_txt):
    images = sorted(images)
    boxes = sorted(boxes)
    scores = []
    for image, box in tqdm.tqdm(zip(images, boxes)):
        name = os.path.basename(image)
        image = cv2.imread(image)
        box = cv2.imread(box, 0)
        box = np.where(box > 128, 255, 0).astype(np.uint8)
        foreground = image[box == 255] / 255.
        background = image[box == 0] / 255.
        if len(foreground) == 0:
            foreground = 1
        if len(background) == 0:
            background = 0
        fg_hist, fg_bins = np.histogram(foreground, bins=30, range=(0, 1), density=True)
        bg_hist, bg_bins = np.histogram(background, bins=30, range=(0, 1), density=True)
        kl_divergence = entropy(fg_hist + 1e-10, bg_hist + 1e-10) 
        m = 0.5 * (fg_hist + bg_hist)
        js_divergence = 0.5 * (entropy(fg_hist + 1e-10, m + 1e-10) + entropy(bg_hist + 1e-10, m + 1e-10))
        scores.append([os.path.basename(name), ':', str(kl_divergence + js_divergence)])
        scores = sorted(scores, key=lambda x: x[2])
    with open(save_path_txt, 'w') as f:
        for score in scores:
            score = ''.join(score)
            f.write(str(score) + '\n')
    return scores   


def MDE_coreset(S_path, alpha, beta, k, path_img, save_path):
    # Load dataset
    with open(S_path, 'r') as f:
        S = f.readlines()

    S = [s.strip().split(':') for s in S]
    S = [(s[0], float(s[1])) for s in S]
    n = len(S)

    # Calculate the number of elements to select
    num = int(n * alpha / k)

    # Step 2: Prune hard examples
    sorted_S = sorted(S, key=lambda x: x[1])
    num_pruned = int(n * beta)
    S_prime = sorted_S[num_pruned:]

    # Split into k subsets
    R = np.array_split(S_prime, k)
    S_c = []

    # Select samples based on maximum entropy
    for R_i in R:
        selected = select_according_max_entropy_torch([i for i, _ in R_i], num, path_img)
        S_c.extend(selected)

    # Save the selected samples to a file
    output_file = save_path + str(int(100 * alpha)) + '.txt'
    with open(output_file, 'w') as f:
        for s in S_c:
            f.write(f"{s}\n")


def wasserstein_distance_torch(x, y):
    """
    Calculate the 1D Wasserstein distance between two distributions.
    """
    # Ensure input is 1D tensor
    x = x.flatten()
    y = y.flatten()

    # Sort both distributions
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)

    # Calculate Wasserstein distance
    return torch.sum(torch.abs(x_sorted - y_sorted)) / len(x)


def select_according_max_entropy_torch(R, num, path_img, device='cuda:2'):
    """
    Select samples from R based on maximum entropy using Wasserstein distance.
    """
    # Load and preprocess images
    image_np = [cv2.imread(path_img + r) for r in R]
    image_np = [cv2.resize(img, (512, 512)).ravel() for img in image_np]

    # Convert image data to PyTorch tensor and move to the specified device (CPU or GPU)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).to(device)

    # Initialize selection
    selected = [R[0]]  # Select the first sample
    reference_distance = image_tensor[0]  # Use the first image as the initial reference
    image_tensor = image_tensor[1:]
    R = R[1:]

    # Iteratively select samples until the desired number is reached
    while len(selected) < num:
        # Calculate Wasserstein distance between the reference and all remaining samples
        distances = [wasserstein_distance_torch(reference_distance, img) for img in image_tensor]
        distances = torch.stack(distances)  # Combine distances into a single tensor

        # Find the sample with the maximum Wasserstein distance
        max_idx = torch.argmax(distances).item()
        selected.append(R[max_idx])

        # Update the reference distance using the mean of the last 5 selected samples
        recent_selected = [img for i, img in enumerate(image_tensor) if R[i] in selected[-5:]]
        selected_tensor = torch.stack(recent_selected)
        reference_distance = torch.mean(selected_tensor, dim=0)

        # Remove the selected sample from the pool
        image_tensor = torch.cat([image_tensor[:max_idx], image_tensor[max_idx + 1:]])
        R = R[:max_idx] + R[max_idx + 1:]

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
    # images = '/mnt/jixie16t/dataset/Polyp/TrainDataset/image'
    # boxes = '/mnt/jixie16t/dataset/Polyp/TrainDataset/mask'
    # scores = get_sim_socre_according_foreground_background(glob.glob(os.path.join(images, '*.jpg')), glob.glob(os.path.join(boxes, '*.png')), '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.2, 0.1, 5, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.01, 0.1, 5, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.05, 0.1, 5, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    # MDE_coreset("/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset_select.txt", 0.1, 0.1, 5, images + '/', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset')
    generate_namelist('/mnt/jixie16t/dataset/Polyp/TrainDataset/image', '/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt')
    remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset20.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining80.txt')
    remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset1.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining99.txt')
    remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset5.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining95.txt')
    remaining_set('/mnt/jixie16t/dataset/Polyp/TrainDataset/namelist.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/coreset10.txt', '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/Polyp/coreset/remaining90.txt')

    remaining_set("/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset20.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining80.txt")
    remaining_set("/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset1.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining99.txt")
    remaining_set("/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset5.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining95.txt")
    remaining_set("/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/coreset10.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/COD/coreset/remaining90.txt")
    
    generate_namelist("/mnt/jixie16t/dataset/DIS5K/DIS-TR/im", "/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt")
    remaining_set("/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset20.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining80.txt")
    remaining_set("/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset1.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining99.txt")
    remaining_set("/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset5.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining95.txt")
    remaining_set("/mnt/jixie16t/dataset/DIS5K/DIS-TR/namelist.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/coreset10.txt", "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/configs/DIS5K/coreset/remaining90.txt")
