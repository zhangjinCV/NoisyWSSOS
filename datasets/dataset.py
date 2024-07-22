import os
import yaml
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import albumentations as A
import json, glob, cv2

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 读取配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 构建数据增强管道
aug = A.Compose([
    A.ColorJitter(0.5, 0.5, 0.5, 0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.GaussNoise(p=0.2),
    A.Blur(p=0.2),
    A.ShiftScaleRotate(rotate_limit=30),
    A.ToGray(p=0.2),
    A.Emboss(p=0.5),
    A.Posterize(p=0.5),
    A.Perspective(p=0.5)
], additional_targets={'image2': 'image', 'mask': 'mask', 'edge': 'mask'})


class LVISDataset(data.Dataset):
    def __init__(self, image_root, gt_root, json_path='/mnt/jixie16t/dataset/LVIS/lvis_v1_train.json', trainsize=384, istraining=True):
        self.trainsize = trainsize
        self.istraining = istraining
        self.image_root = image_root
        self.gt_root = gt_root
        
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = glob.glob(image_root + '/*.jpg')
        self.gts = [i.replace(image_root, gt_root).replace('.jpg', '.png').replace("_coconut", "") for i in self.images]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        name = os.path.basename(self.images[index])
        gt = cv2.imread(self.gts[index], 0)
        gt = self.process_mask(gt)
        bbox = self.get_bbox(gt)
        bbox_image = self.apply_bbox_to_image(image, bbox)

        H, W = image.size
        if self.istraining:
            image, bbox_image, gt = np.array(image).astype(np.uint8), np.array(bbox_image).astype(
                np.uint8), np.array(gt).astype(np.uint8)
            augmented = aug(image=image, image2=bbox_image, mask=gt)
            image, bbox_image, gt = augmented['image'], augmented['image2'], augmented['mask']
            image, bbox_image, gt = Image.fromarray(image), Image.fromarray(bbox_image), Image.fromarray(gt)
        image = self.img_transform(image)
        bbox_image = self.img_transform(bbox_image)
        gt = self.gt_transform(gt)
        return {'image': image, 'bbox_image': bbox_image, 'gt': gt, 'H': H, 'W': W, 'name': name}

    def process_mask(self, mask):
        # Assuming mask contains multiple classes, we select a random class
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]  # Remove background class if it is 0
        selected_class = np.random.choice(unique_classes)  # Randomly select a class
        binary_mask = (mask == selected_class).astype(np.uint8) * 255
        return Image.fromarray(binary_mask)

    def get_bbox(self, binary_mask):
        binary_array = np.array(binary_mask)
        pos = np.where(binary_array == 255)
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        return (xmin, ymin, xmax, ymax)

    def apply_bbox_to_image(self, image, bbox):
        xmin, ymin, xmax, ymax = bbox
        image_array = np.array(image)
        masked_image = np.zeros_like(image_array)
        masked_image[ymin:ymax, xmin:xmax, :] = image_array[ymin:ymax, xmin:xmax, :]
        return Image.fromarray(masked_image)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class COSwithNoBox(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, istraining=True, repeat=1):
        self.trainsize = trainsize
        self.istraining = istraining
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')] * repeat
        self.gts = [i.replace("image", "mask").replace(".jpg", ".png") for i in self.images] * repeat
        self.edges = [i.replace("mask", "edge") for i in self.gts] * repeat

        if self.istraining:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))
            self.edges = np.array(sorted(self.edges))
        else:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        name = os.path.basename(self.images[index])
        gt = self.binary_loader(self.gts[index])
        H, W = image.size
        if self.istraining:
            edge = self.binary_loader(self.edges[index])
            image, gt, edge = np.array(image).astype(np.uint8), np.array(gt).astype(np.uint8), np.array(edge).astype(np.uint8)
            augmented = aug(image=image, mask=gt, edge=edge)
            image, gt, edge = augmented['image'], augmented['mask'], augmented['edge']
            gt = np.where(gt > 0.2 * 255, 255, 0).astype(np.uint8)
            edge = np.where(edge > 0.2 * 255, 255, 0).astype(np.uint8)
            image, gt, edge = Image.fromarray(image),  Image.fromarray(gt), Image.fromarray(edge)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.istraining:
            edge = self.gt_transform(edge)
            return {'image': image, 'gt': gt, 'edge': edge, 'H': H, 'W': W}
        else:
            return {'image': image, 'gt': gt, 'H': H, 'W': W, 'name': name}

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class COSwithBox(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, istraining=True):
        self.trainsize = trainsize
        self.istraining = istraining
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.bbox_gts = [i.replace("mask", "box") for i in self.gts]
        self.edges = [i.replace("mask", "edge") for i in self.gts]

        self.check_pairs(self.images, self.gts, self.bbox_gts, self.edges)
        
        if self.istraining:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))
            self.bbox_gts = np.array(sorted(self.bbox_gts))
            self.edges = np.array(sorted(self.edges))
        else:
            self.images = np.array(sorted(self.images))
            self.gts = np.array(sorted(self.gts))
            self.bbox_gts = np.array(sorted(self.bbox_gts))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        name = os.path.basename(self.images[index])
        gt = self.binary_loader(self.gts[index])
        bbox = self.rgb_loader(self.bbox_gts[index])
        bbox_image = np.array(bbox) / 255. * np.array(image)
        bbox_image = Image.fromarray(bbox_image.astype(np.uint8))
        H, W = image.size
        if self.istraining:
            edge = self.binary_loader(self.edges[index])
            image, bbox_image, gt, edge = np.array(image).astype(np.uint8), np.array(bbox_image).astype(
                np.uint8), np.array(gt).astype(np.uint8), np.array(edge).astype(np.uint8)
            augmented = aug(image=image, image2=bbox_image, mask=gt, edge=edge)
            image, bbox_image, gt, edge = augmented['image'], augmented['image2'], augmented['mask'], augmented['edge']
            image, bbox_image, gt, edge = Image.fromarray(image), Image.fromarray(bbox_image), Image.fromarray(
                gt), Image.fromarray(edge)
        image = self.img_transform(image)
        bbox_image = self.img_transform(bbox_image)
        gt = self.gt_transform(gt)
        if self.istraining:
            edge = self.gt_transform(edge)
            return {'image': image, 'bbox_image': bbox_image, 'gt': gt, 'edge': edge, 'H': H, 'W': W}
        else:
            return {'image': image, 'bbox_image': bbox_image, 'gt': gt, 'H': H, 'W': W, 'name': name}

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def check_pairs(self, images, gts, bbox_gts, edges):
        print("checking dataset pairs")
        assert len(images) == len(gts) == len(bbox_gts) == len(edges)
        print("check dataset pass!")
    def __len__(self):
        return self.size

def get_dataset(config, dataset_key):
    dataset_config = config['dataset'][dataset_key]
    dataset_class = globals()[dataset_config['type']]
    dataset = dataset_class(
        image_root=dataset_config['image_root'],
        gt_root=dataset_config['gt_root'],
        trainsize=dataset_config['trainsize'],
        istraining=dataset_config['istraining'],
        repeat=dataset_config.get('repeat', 1)
    )
    return dataset