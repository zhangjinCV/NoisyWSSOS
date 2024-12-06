import os
import yaml
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import albumentations as A
from pycocotools import mask as ppmask
import json, glob, cv2, tqdm 
from torch.nn import functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import copy

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def decode_segmentation(rle, height, width):
    """Decode RLE segmentation to a binary mask."""
    return ppmask.decode({
        'size': [height, width],
        'counts': rle
    })

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
], additional_targets={'image2': 'image', 'mask': 'mask', 'edge': 'mask', 'mask2': 'mask', 'mask3': 'mask'})


def read_filenames(txt_file):
    with open(txt_file, 'r') as f:
        filenames = f.read().splitlines()
    return filenames


class SA1B(data.Dataset):
    def __init__(self, image_root, json_root, file_list, trainsize, istraining=True, repeat=1, **kwargs):
        self.trainsize = trainsize
        self.istraining = istraining
        self.json_files = glob.glob(os.path.join(json_root, '*.json'))  
        self.images = [glob.glob(os.path.join(image_root, '*.jpg'))]
        common = set([os.path.basename(i).replace('.jpg', '') for i in self.images[0]]) & set([os.path.basename(i).replace('.json', '') for i in self.json_files])
        self.images = [os.path.join(image_root, i + '.jpg') for i in common]
        self.json_files = [os.path.join(json_root, i + '.json') for i in common]

        self.json_files = np.array(sorted(self.json_files))
        self.images = np.array(sorted(self.images))

        self.check_pairs(self.images, self.json_files)
        
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
        json_path = self.json_files[index]

        # Load JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        maxs = len(annotations)
        min_choice = min(30, maxs)
        selected_annotations = random.sample(annotations, random.randint(1, min_choice))

        width, height = image.size
        mask_image = np.zeros((height, width), dtype=np.uint8)
        bbox_image = np.zeros((height, width, 3), dtype=np.uint8)

        for annotation in selected_annotations:
            rle = annotation['segmentation']['counts']
            binary_mask = self.decode_segmentation(rle, height, width)
            mask_image = np.maximum(mask_image, binary_mask)

            bbox = annotation['bbox']
            x, y, w, h = map(int, bbox)
            bbox_image[y:y+h, x:x+w, :] = np.array(image)[y:y+h, x:x+w, :]

        # Convert mask to 255 scale
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image)
        bbox_pil = Image.fromarray(bbox_image)

        # Create edge map using max pooling
        mask_tensor = torch.from_numpy(mask_image).unsqueeze(0).unsqueeze(0).float() / 255.0  # Convert to a torch tensor
        edge_map = F.max_pool2d(mask_tensor, kernel_size=5, stride=1, padding=2) - mask_tensor
        edge_map = (edge_map.squeeze().numpy() * 255).astype(np.uint8)  # Convert back to numpy and scale to 255

        edge_pil = Image.fromarray(edge_map)

        H, W = image.size
        if self.istraining:
            image, bbox_pil, mask_pil, edge_pil = np.array(image).astype(np.uint8), np.array(bbox_pil).astype(
                np.uint8), np.array(mask_pil).astype(np.uint8), np.array(edge_pil).astype(np.uint8)
            augmented = aug(image=image, image2=bbox_pil, mask=mask_pil, edge=edge_pil)
            image, bbox_pil, mask_pil, edge_pil = augmented['image'], augmented['image2'], augmented['mask'], augmented['edge']
            image, bbox_pil, mask_pil, edge_pil = Image.fromarray(image), Image.fromarray(bbox_pil), Image.fromarray(mask_pil), Image.fromarray(edge_pil)

        image = self.img_transform(image)
        bbox_image = self.img_transform(bbox_pil)
        mask_image = self.gt_transform(mask_pil)
        edge_image = self.gt_transform(edge_pil)

        return {'image': image, 'bbox_image': bbox_image, 'gt': mask_image, 'edge': edge_image, 'H': H, 'W': W, 'name': name}

    def rgb_loader(self, path):
        img = Image.open(path)
        return img

    def decode_segmentation(self, rle, height, width):
        """Decode RLE segmentation to a binary mask."""
        return ppmask.decode({
            'size': [height, width],
            'counts': rle
        })

    def check_pairs(self, images, json_files):
        print("checking dataset pairs")
        assert len(images) == len(json_files)
        print("check dataset pass!")

    def __len__(self):
        return self.size
    

class LVISDataset(data.Dataset):
    def __init__(self, image_root, gt_root, file_list, json_path='/mnt/jixie16t/dataset/LVIS/lvis_v1_train.json', trainsize=384, istraining=True):
        self.trainsize = trainsize
        self.istraining = istraining
        self.image_root = image_root
        self.gt_root = gt_root
        
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = [os.path.join(image_root, fname) for fname in read_filenames(file_list)]
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
        else:
            image, bbox_image, gt = np.array(image).astype(np.uint8), np.array(bbox_image).astype(
                np.uint8), np.array(gt).astype(np.uint8)
            augmented = aug2(image=image, image2=bbox_image, mask=gt)
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
    def __init__(self, image_root, gt_root, file_list, trainsize, istraining=True, repeat=1, **kwargs):
        self.trainsize = trainsize
        self.istraining = istraining
        if 'edge_root' in kwargs:
            edge_root = kwargs['edge_root']
        else:
            print("you have not set the edge_root in yaml file, so we would not use the edge, instand of the mask path")
            edge_root = gt_root
        self.images, self.gts, self.edges = self.handle_read_path(image_root, gt_root, edge_root, file_list, repeat)
        print("len(self.images):", len(self.images), "len(self.gts):", len(self.gts), "len(self.edges):", len(self.edges))
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
            image, gt, edge = Image.fromarray(image),  Image.fromarray(gt), Image.fromarray(edge)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.istraining:
            edge = self.gt_transform(edge)
            return {'image': image, 'gt': gt, 'edge': edge, 'H': H, 'W': W, 'name': name}
        else:
            return {'image': image, 'gt': gt, 'H': H, 'W': W, 'name': name}

    def handle_read_path(self, image_roots, gt_roots, edge_roots, file_lists, repeats, **kwargs):
        if isinstance(image_roots, str):
            self.images = [os.path.join(image_roots, fname) for fname in read_filenames(file_lists)] * repeats
            self.gts = [os.path.join(gt_roots, fname.replace('.jpg', '.png')) for fname in read_filenames(file_lists)] * repeats
            self.edges = [os.path.join(edge_roots, fname.replace('.jpg', '.png')) for fname in read_filenames(file_lists)] * repeats
        else:
            assert isinstance(image_roots, list)
            self.images = []
            self.gts = []
            self.edges = []
            for image_root, gt_root, edge_root, file_list, repeat in zip(image_roots, gt_roots, edge_roots, file_lists, repeats):
                self.images += [os.path.join(image_root, fname) for fname in read_filenames(file_list)] * repeat
                self.gts += [os.path.join(gt_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)] * repeat 
                self.edges += [os.path.join(edge_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)] * repeat 
                print(image_root, gt_root, edge_root, file_list, repeat)
                print(len(self.images), len(self.gts))
            
        return np.array(self.images), np.array(self.gts), np.array(self.edges)


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def check_pairs(self, images, gts, edges):
        print("checking dataset pairs")
        if self.istraining:
            for img, gt, edge in tqdm.tqdm(zip(images, gts, edges)):
                assert os.path.exists(img), f"Image file {img} does not exist"
                assert os.path.exists(gt), f"GT file {gt} does not exist"
                assert os.path.exists(edge), f"Edge file {edge} does not exist"
        else:
            for img, gt in tqdm.tqdm(zip(images, gts)):
                assert os.path.exists(img), f"Image file {img} does not exist"
                assert os.path.exists(gt), f"GT file {gt} does not exist"
        print("check dataset pass!")

    def __len__(self):
        return self.size


class COSwithBox(data.Dataset):
    def __init__(self, image_root, gt_root, file_list, trainsize, istraining=True, repeat=1, **kwargs):
        self.trainsize = trainsize
        self.istraining = istraining
        self.gts = [os.path.join(gt_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)]
        self.images = [os.path.join(image_root, fname) for fname in read_filenames(file_list)]
        if 'box_root' in kwargs:
            box_root = kwargs['box_root']
            self.bbox_gts = [os.path.join(box_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)]
        else:
            self.bbox_gts = [i.replace("mask", "box") for i in self.gts]
        if self.istraining:
            self.edges = [os.path.join(kwargs['edge_root'], fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)]
            self.check_pairs(self.images, self.gts, self.bbox_gts, self.edges)
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
        self.bbox_resize_aug = A.Compose([
            A.Affine(scale=(0.9, 1.1)),  # Resize bounding box with scaling between 0.8x to 1.2x
        ], additional_targets={'bbox_image': 'mask'})

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        H, W = image.size
        name = os.path.basename(self.images[index])
        gt = self.binary_loader(self.gts[index])
        bbox = self.rgb_loader(self.bbox_gts[index])
        if self.istraining:
            bbox = self.bbox_resize_aug(image=np.array(image), bbox_image=np.array(bbox))['bbox_image']
        else:
            bbox = np.array(bbox)
        bbox = cv2.resize(bbox, (H, W))
        bbox_image = np.array(bbox) / 255. * np.array(image)
        bbox_image = Image.fromarray(bbox_image.astype(np.uint8))
        
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
            return {'image': image, 'bbox_image': bbox_image, 'gt': gt, 'edge': edge, 'H': H, 'W': W, 'name': name}
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


class COSwithBoxVal(data.Dataset):
    def __init__(self, image_root, gt_root, file_list, trainsize, istraining=True, repeat=1, **kwargs):
        self.trainsize = trainsize
        self.istraining = istraining
        self.gts = [os.path.join(gt_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)]
        self.images = [os.path.join(image_root, fname) for fname in read_filenames(file_list)]
        if 'box_root' in kwargs:
            box_root = kwargs['box_root']
            self.bbox_gts = [os.path.join(box_root, fname.replace('.jpg', '.png')) for fname in read_filenames(file_list)]
        else:
            self.bbox_gts = [i.replace("mask", "box") for i in self.gts]
        self.edges = [i.replace("mask", "edge") for i in self.gts]

        self.check_pairs(self.images, self.gts, self.bbox_gts, self.edges)
        
        self.images = np.array(sorted(self.images))
        self.gts = np.array(sorted(self.gts))
        self.bbox_gts = np.array(sorted(self.bbox_gts))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        H, W = image.size
        name = os.path.basename(self.images[index])
        gt = self.binary_loader(self.gts[index])
        bbox = self.rgb_loader(self.bbox_gts[index])
        bbox = np.array(bbox)
        bbox = cv2.resize(bbox, (H, W))
        bbox_image = np.array(bbox) / 255. * np.array(image)
        bbox_image = Image.fromarray(bbox_image.astype(np.uint8))
        image = self.img_transform(image)
        bbox_image = self.img_transform(bbox_image)
        gt = self.gt_transform(gt)
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
        **dataset_config
        # image_root=dataset_config['image_root'],
        # gt_root=dataset_config['gt_root'],
        # file_list=dataset_config['file_list'],
        # trainsize=dataset_config['trainsize'],
        # istraining=dataset_config['istraining'],
        # repeat=dataset_config.get('repeat', 1)
    )
    return dataset  
