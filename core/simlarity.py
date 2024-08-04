import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from glob import glob
import tqdm, json

def read_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

def extract_foreground(image, mask):
    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground

def split_image(image, num_splits):
    h, w = image.shape[:2]
    split_h, split_w = h // num_splits, w // num_splits
    sub_images = []
    for i in range(num_splits):
        for j in range(num_splits):
            sub_img = image[i * split_h:(i + 1) * split_h, j * split_w:(j + 1) * split_w]
            sub_images.append(sub_img)
    return sub_images

def calculate_similarity(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    similarity, _ = ssim(gray_image1, gray_image2, full=True)
    return similarity

def main(image_dir, mask_dir, num_splits=4):
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))

    similarities = []

    for image_path, mask_path in tqdm.tqdm(zip(image_paths, mask_paths)):
        image, mask = read_image_and_mask(image_path, mask_path)
        foreground = extract_foreground(image, mask)
        background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

        fg_bg_similarities = [calculate_similarity(foreground, background)]

        total_similarity = np.mean(fg_bg_similarities)
        similarities.append((image_path, total_similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def main2(image_dir, box_dir, image2_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
    box_paths = sorted(glob(os.path.join(box_dir, '*.png')))
    image2_paths = sorted(glob(os.path.join(image2_dir, '*.png')))

    similarities = []

    for image_path, box_path, image2_path in tqdm.tqdm(zip(image_paths, box_paths, image2_paths)):
        image, box = read_image_and_mask(image_path, box_path)
        image2, _ = read_image_and_mask(image2_path, box_path)

        masked_image1 = extract_foreground(image, box)
        masked_image2 = extract_foreground(image2, box)

        similarity = 1 - calculate_similarity(masked_image1, masked_image2)
        similarities.append((image_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=False)
    return similarities

def combine_and_sort_similarities(sim1, sim2):
    res = {}
    for path1, sim1_val in sim1:
        base_name = os.path.basename(path1)
        res[base_name] = sim1_val * 0.3
    for path2, sim2_val in sim2:
        base_name = os.path.basename(path2)
        res[base_name] += sim2_val * 0.7 
    return res

def save_top_20_percent(sorted_res, save_path):
    num_top = int(len(sorted_res))
    top_20_percent = sorted(sorted_res.items(), key=lambda item: item[1], reverse=True)[:num_top]

    with open(save_path, 'w') as f:
        json.dump(top_20_percent, f)


def readjson2txt(json_path, top_txt_path, remaining_txt_path):
    # Read JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    top_20_percent_index = int(len(sorted_data) * 0.2)
    
    top_20_percent_images = [item[0] for item in sorted_data[:top_20_percent_index]]
    remaining_80_percent_images = [item[0] for item in sorted_data[top_20_percent_index:]]
    
    with open(top_txt_path, 'w') as file:
        for image_name in top_20_percent_images:
            file.write(f"{image_name}\n")
    
    with open(remaining_txt_path, 'w') as file:
        for image_name in remaining_80_percent_images:
            file.write(f"{image_name}\n")


if __name__ == "__main__":
    image_dir = "/mnt/jixie16t/dataset/COD/CAMO_COD_train/image"
    mask_dir = "/mnt/jixie16t/dataset/COD/CAMO_COD_train/box"
    image2_dir = "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/data/ANet/ImageInpainting/mask"
    save_path = "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/data/similarities.json"

    # sim2 = main2(image_dir, mask_dir, image2_dir)
    # sim1 = main(image_dir, mask_dir, num_splits=1)
    
    # res = combine_and_sort_similarities(sim1, sim2)
    # save_top_20_percent(res, save_path)

    readjson2txt(save_path,
        '/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/top_20_percent.txt',
        '/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/remaining_80_percent.txt'
    )