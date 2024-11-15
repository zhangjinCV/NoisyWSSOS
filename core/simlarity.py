import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from glob import glob
import tqdm, json, random

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


def cal_hard_rate(path):
    with open('/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/similarities.json', 'r') as file:
        data = json.load(file)
    
    data_list = []
    with open(path, 'r') as file:
        for line in file:
            data_list.append(line.strip().replace(".jpg", ".png"))
    hard_rate = []
    for item in tqdm.tqdm(data_list):
        hard_rate.append(data[os.path.join("/mnt/jixie16t/dataset/COD/CAMO_COD_train/mask", item)])
    print(np.mean(hard_rate))
    

def select_methods():
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/similarities.json", 'r') as f:
        sim = json.load(f)
    key = list(sim.keys())
    values = list(sim.values())
    len_key = len(key)
    # 在value区间[0.5-0.8]选择20%的数据
    select_20 = []
    for i in range(len_key):
        if values[i] >= 0.6 and values[i] <= 1.0:
            select_20.append(key[i])
    random.shuffle(select_20)
    select_20 = select_20[:int(len_key * 0.3)]
    print(len(select_20))
    select_hard = []
    for i in range(len_key):
        if values[i] >= 0.0 and values[i] <= 0.3:
            select_hard.append(key[i])
    random.shuffle(select_hard)
    select_hard = select_hard[:int(len_key * 0.1)]
    select_20 = select_20 + select_hard
    print(len(select_20))
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/hard0.4.txt", 'w') as f:
        for one in select_20:
            f.write(os.path.basename(one).replace("png", "jpg") + '\n')


def cal_diff_l1(image_groups, target_image):
    # 使用NumPy的广播来计算差异
    diffs = np.mean(np.abs(image_groups - target_image), axis=(1, 2, 3))
    return diffs

def preprocess_images(image_paths):
    # 预处理所有图像，调整大小并转换为NumPy数组
    images = [cv2.resize(cv2.imread(path.replace("mask", "image").replace("png", "jpg")), (384, 384)) for path in image_paths]
    return np.array(images)


def load_and_resize_images(keys):
    images = {}
    for key in keys:
        path = key.replace("mask", "image").replace("png", "jpg")
        images[key] = cv2.resize(cv2.imread(path), (224, 224))
    return images

def select_images(image_dict, keys, target_num):
    selected = {}
    remaining_keys = keys[:]
    if keys:
        selected[keys[0]] = image_dict[keys[0]]
        remaining_keys.pop(0)
    
    while len(selected) < target_num:
        max_diff = 0
        index = -1
        current_mean_image = np.mean(list(selected.values()), axis=0)
        
        for i, key in enumerate(remaining_keys):
            diff = np.mean(np.abs(image_dict[key] - current_mean_image))
            if diff > max_diff:
                max_diff = diff
                index = i
        
        if index != -1:
            selected[remaining_keys[index]] = image_dict[remaining_keys[index]]
            remaining_keys.pop(index)
        
        if len(selected) % 100 == 0:
            print(f"Selected {len(selected)} images.")
    
    return selected

def select_methods2():
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/similarities.json", 'r') as f:
        sim = json.load(f)

    keys = list(sim.keys())
    image_dict = load_and_resize_images(keys)

    keys_hard = [k for k, v in sim.items() if 0.7 <= v <= 1.0]
    selected_hard = select_images(image_dict, keys_hard, 505 * 2)

    keys_easy = [k for k, v in sim.items() if 0.0 <= v <= 0.4]
    selected_easy = select_images(image_dict, keys_easy, 303 * 2)

    all_selected = list(selected_hard.keys()) + list(selected_easy.keys())
    
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/hard0.4.txt", 'w') as f:
        for one in all_selected:
            f.write(os.path.basename(one).replace("png", "jpg") + '\n')

def cal_iou(mask1, mask2):
    # 计算两者的iou

    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    iou_score = intersection / (union + 1e-12)
    return iou_score

def main3(image1_dir, image2_dir):
    from PIL import Image
    name = os.listdir(image1_dir)
    similarities = []
    # Calculate similarities and store them in a list of tuples
    for path in tqdm.tqdm(name):
        image_path = os.path.join(image1_dir, path)
        mask_path = os.path.join(image2_dir, path.replace(".png", ".png"))
        img1 = cv2.imread(image_path, 0) / 255.
        img2 = cv2.imread(mask_path, 0) / 255.
        similarity = cal_iou(img1, img2)
        similarities.append((image_path, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Save the similarities to a JSON file
    similarities_dict = {image_path: similarity for image_path, similarity in similarities}
    with open('/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/UNet/similarities.json', 'w') as f:
        json.dump(similarities_dict, f)

    # # Proportional extraction of 20% of the paths
    # image_path_sorted = [image_path for image_path, _ in similarities]
    # seect_image_path = image_path_sorted[::5]

    # # Save the selected paths to a text file
    # with open('/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/pruning_methods/namelist_pot_0.2.txt', 'w') as f:
    #     for path in  seect_image_path:
    #         f.write(f"{path}\n")

    return similarities_dict


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

    similarities = {}

    for image_path, box_path, image2_path in tqdm.tqdm(zip(image_paths, box_paths, image2_paths)):
        image, box = read_image_and_mask(image_path, box_path)
        image2, _ = read_image_and_mask(image2_path, box_path)

        masked_image1 = extract_foreground(image, box)
        masked_image2 = extract_foreground(image2, box)

        similarity = 1 - calculate_similarity(masked_image1, masked_image2)
        similarities[image_path] = similarity

    similarities.sort(key=lambda x: x[1], reverse=False)
    return similarities


def select_20_percent():
    with open('/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/pruning_methods/similarities.json', 'r+') as f:
        data = json.load(f)
    # 间隔采样20%，只保存image name
    print(data)


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
    
    sorted_data = sorted(data, key=lambda x: x[1], reverse=False)
    
    top_20_percent_images = []
    remaining_80_percent_images = []
    for i in range(len(sorted_data)):
        if i % 5 == 0:
            top_20_percent_images.append(sorted_data[i][0])
        else:
            remaining_80_percent_images.append(sorted_data[i][0])

    # top_20_percent_index = int(len(sorted_data) * 0.2)
    
    # top_20_percent_images = [item[0] for item in sorted_data[:top_20_percent_index]]
    # remaining_80_percent_images = [item[0] for item in sorted_data[top_20_percent_index:]]
    
    with open(top_txt_path, 'w') as file:
        for image_name in top_20_percent_images:
            file.write(f"{image_name}\n")
    
    with open(remaining_txt_path, 'w') as file:
        for image_name in remaining_80_percent_images:
            file.write(f"{image_name}\n")

































from sklearn.metrics.pairwise import cosine_similarity

def cal_cosine_similarity(imagePath1, imagePath2, imagePath3, save_path):
    all_cosine_similarities = []
    
    images1 = sorted(glob(os.path.join(imagePath1, '*')))
    images2 = sorted(glob(os.path.join(imagePath2, '*')))
    images3 = sorted(glob(os.path.join(imagePath3, '*')))
    names = os.listdir(imagePath1)
    names = [i.replace(".JPEG", "") for i in names]
    for name in tqdm.tqdm(names):
        img1 = os.path.join(imagePath1, name + ".JPEG")  # 读取并调整图像尺寸
        img2 = os.path.join(imagePath2, name + ".JPEG")
        img3 = os.path.join(imagePath3, name + ".png")
        
        # 读取并调整图像尺寸
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2)
        image3 = cv2.imread(img3)
        brokend_area = image3

        if np.sum(brokend_area) == 0:
            continue
        
        # 计算破损区域的掩码
                
        # 确保图像是 uint8 类型
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)
        
        # 应用掩码
        # brokend_img1 = image1 * brokend_area[:, :, np.newaxis]
        # brokend_img2 = image2 * brokend_area[:, :, np.newaxis]
        brokend_img1 = image1.reshape(-1)
        brokend_img2 = image2.reshape(-1)
        brokend_area = brokend_area.reshape(-1)
        brokend_img1 = brokend_img1[brokend_area != 0]
        brokend_img2 = brokend_img2[brokend_area != 0]
        cosine_sim = np.mean(np.abs(brokend_img1 - brokend_img2)) / 255.
        all_cosine_similarities.append([name, cosine_sim])
    
    # 按照余弦相似度从大到小排序
    all_cosine_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 保存到json文件
    with open(save_path, 'w') as f:
        json.dump(all_cosine_similarities, f, ensure_ascii=False, indent=4)


def max_l1_select_methods(existing_group, remaining_group, target_count):
    existing_group_np = []
    remaining_group_np = []
    import torch
    for i in existing_group:
        im = cv2.imread(i)
        im = cv2.resize(im, (224, 224))
        existing_group_np.append(im)

    for i in remaining_group:
        im = cv2.imread(i)
        im = cv2.resize(im, (224, 224))
        remaining_group_np.append(im)

    existing_group_np = np.array(existing_group_np)
    remaining_group_np = np.array(remaining_group_np)
    print(existing_group_np.shape, remaining_group_np.shape)
    existing_group_np = torch.from_numpy(existing_group_np).float().cuda()
    remaining_group_np = torch.from_numpy(remaining_group_np).float().cuda()

    def select_one(existing_group_np, remaining_group_np, existing_group, remaining_group):
        batch_size = 20  # 一次计算10个
        num_remaining = remaining_group_np.shape[0]  # 剩余图像的数量
        l1_distances = []
        
        # 分批计算 L1 距离
        for i in range(0, num_remaining, batch_size):
            # 取出每批的图像, 如果不足batch_size就直接取剩下的
            batch = remaining_group_np[i:i + batch_size]
            # 计算该批次和 existing_group_np 的 L1 距离
            batch_distances = torch.abs(batch.unsqueeze(1) - existing_group_np).mean(dim=(1, 2, 3, 4))
            # 存储该批次的距离
            l1_distances.append(batch_distances)

        # 将所有批次的距离拼接成一个整体
        l1_distances = torch.cat(l1_distances, dim=0).cuda()
        selected_index = torch.argmax(l1_distances).item()
        existing_group_np = torch.mean(torch.cat([existing_group_np, remaining_group_np[selected_index].unsqueeze(0)], dim=0), dim=0, keepdim=True)
        remaining_group_np = torch.cat([remaining_group_np[:selected_index], remaining_group_np[selected_index+1:]], dim=0)
        existing_group.append(remaining_group[selected_index])
        remaining_group.pop(selected_index)
        return existing_group_np, remaining_group_np, existing_group, remaining_group

    while len(existing_group) < target_count and len(remaining_group_np) > 0:
        torch.cuda.empty_cache()
        existing_group_np, remaining_group_np, existing_group, remaining_group = select_one(existing_group_np, remaining_group_np, existing_group, remaining_group)
        if len(existing_group) % 10 == 0:
            print(f"Selected {len(existing_group)} images.")

    return existing_group  # 返回列表形式的现有组


def pot_select_methods():
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/similarities.json",
              'r') as f:
        pot_values = json.load(f)
    for i in range(len(pot_values)):
        pot_values[i][0] = '/mnt/jixie16t/dataset/imagenet/imagenet-s/train/image/' + pot_values[i][0] + '.JPEG'
    # 统计values的分布
    # import matplotlib.pyplot as plt
    # plt.hist(img_values, bins=20)
    # plt.savefig("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/hist.png")
    rations = [0.9, 0.1, 0]
    values_less_than_0_3 = [i for i in pot_values if i[1] <= 0.65]
    values_greater_than_0_7 = [i for i in pot_values if i[1] >= 0.65]
    values_between_0_3_and_0_7 = [i for i in pot_values if i[1] > 0.4 and i[1] < 0.6]

    values_less_than_0_3 = [i[0] for i in values_less_than_0_3]
    values_greater_than_0_7 = [i[0] for i in values_greater_than_0_7]
    values_between_0_3_and_0_7 = [i[0] for i in values_between_0_3_and_0_7]

    first_selected = [values_less_than_0_3[0]]
    second_selected = [values_greater_than_0_7[0]]
    third_selected = [values_between_0_3_and_0_7[0]]
 
    values_less_than_0_3.pop(0)
    values_greater_than_0_7.pop(0)
    values_between_0_3_and_0_7.pop(0)
    print(len(values_less_than_0_3), len(values_greater_than_0_7), len(values_between_0_3_and_0_7))
    print(len(pot_values) * rations[0] * 0.2, len(pot_values) * rations[1] * 0.2, len(pot_values) * rations[2] * 0.2)
    assert len(values_less_than_0_3) >= len(pot_values) * rations[0] * 0.2
    assert len(values_greater_than_0_7) >= len(pot_values) * rations[1] * 0.2
    assert len(values_between_0_3_and_0_7) >= len(pot_values) * rations[2] * 0.2

    sel1 = max_l1_select_methods(first_selected, values_less_than_0_3, len(pot_values) * rations[0] * 0.2)
    sel2 = max_l1_select_methods(second_selected, values_greater_than_0_7, len(pot_values) * rations[1] * 0.2)
    sel3 = max_l1_select_methods(third_selected, values_between_0_3_and_0_7, len(pot_values) * rations[2] * 0.2)

    selected_imgs = sel1 + sel2 + sel3
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/pot0.2.txt", 'w') as f:
        for i in selected_imgs:
            f.write(os.path.basename(i) + '\n')


def random_select():
    random.seed(0)
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/similarities.json",
              'r') as f:
        pot_values = json.load(f)
    for i in range(len(pot_values)):
        pot_values[i][0] = '/mnt/jixie16t/dataset/imagenet/imagenet-s/train/image/' + pot_values[i][0] + '.JPEG'
    rations = [0.3, 0.2, 0.5]
    values_less_than_0_3 = [i for i in pot_values if i[1] <= 0.3]
    values_greater_than_0_7 = [i for i in pot_values if i[1] >= 0.7]
    values_between_0_3_and_0_7 = [i for i in pot_values if i[1] > 0.3 and i[1] < 0.7]

    nums = [int(len(pot_values) * rations[0] * 0.2), int(len(pot_values) * rations[1] * 0.2), int(len(pot_values) * rations[2]) * 0.2]
    values_less_than_0_3 = [i[0] for i in values_less_than_0_3]
    chioce_num1 = min(nums[0], len(values_less_than_0_3))
    print(chioce_num1, len(values_less_than_0_3))
    values_less_than_0_3 = random.sample(values_less_than_0_3, chioce_num1)
    values_greater_than_0_7 = [i[0] for i in values_greater_than_0_7]
    chioce_num2 = min(nums[1], len(values_greater_than_0_7))
    values_greater_than_0_7 = random.sample(values_greater_than_0_7, chioce_num2)
    values_between_0_3_and_0_7 = [i[0] for i in values_between_0_3_and_0_7]
    values_between_0_3_and_0_7 = random.sample(values_between_0_3_and_0_7, int(len(pot_values) * 0.2 - chioce_num1 - chioce_num2))
    selected_imgs = values_less_than_0_3 + values_greater_than_0_7 + values_between_0_3_and_0_7
    with open("/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/pot0.2.txt", 'w') as f:
        for i in selected_imgs:
            f.write(os.path.basename(i) + '\n')



if __name__ == "__main__":
    image1_dir = "/mnt/jixie16t/dataset/imagenet/imagenet-s/train/image"
    image2_dir = "/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/results/imagenet-s/pruning_methods_POTLoss/rec_img"
    image3_dir = "/mnt/jixie16t/dataset/imagenet/imagenet-s/train/box"
    save_path = "/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/ImageNet-S/pruning_methods/similarities.json"
    # select_methods2()
    # cal_cosine_similarity(image1_dir, image2_dir, image3_dir, save_path)
    random_select()
    # cal_hard_rate('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist_0.2.txt')
    # cal_hard_rate('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist_0.4.txt')
    # cal_hard_rate('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist_0.6.txt')
    # cal_hard_rate('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist_0.8.txt')
    # cal_hard_rate('/mnt/jixie16t/zj/zj/works_in_phd/DataPruning/configs/COD/baseline/ConvNeXtUNet/hard0.2.txt')
    # cal_hard_rate('/mnt/jixie16t/dataset/COD/CAMO_COD_train/namelist.txt')
    # sim2 = main2(image_dir, mask_dir, image2_dir)
    # sim1 = main(image_dir, mask_dir, num_splits=1)
    
    # res = combine_and_sort_similarities(sim1, sim2)
    # save_top_20_percent(res, save_path)

    # readjson2txt(save_path,
    #     '/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/middle_20_percent.txt',
    #     '/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/remaining_middle_80_percent.txt'
    # )