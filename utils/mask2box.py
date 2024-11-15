import cv2
import numpy as np
import glob
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_counter(img, thread=0.5):
    H_, W_ = img.shape
    img = np.where(img > thread, 1, 0)
    H = np.max(img, axis=0)
    W = np.max(img, axis=1)
    indexs_H = np.argwhere(H == 1)
    indexs_W = np.argwhere(W == 1)
    h_begin = indexs_H[0][0]
    h_end = indexs_H[-1][0]
    w_begin = indexs_W[0][0]
    w_end = indexs_W[-1][0]
    return (h_begin, w_begin, h_end, w_end)

def annotate_image_with_boxes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    image_shape = binary_image.shape
    blank_image = np.zeros(image_shape, dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        blank_image[y:y + h, x:x + w] = 255
    return blank_image

def process_image(img_path):
    box_path = "/mnt/jixie16t/dataset/DIS5K/DIS-TR/box/"
    img = cv2.imread(img_path, 0).astype(np.uint8)
    box = annotate_image_with_boxes(img)
    box = cv2.resize(box, dsize=img.shape[0:2][::-1], interpolation=cv2.INTER_LINEAR)
    save_path = os.path.join(box_path, os.path.basename(img_path))
    cv2.imwrite(save_path, box)

def main():
    imgs_path = glob.glob('/mnt/jixie16t/dataset/DIS5K/DIS-TR/gt/*')
    box_path = "/mnt/jixie16t/dataset/DIS5K/DIS-TR/box/"
    if not os.path.exists(box_path):
        os.makedirs(box_path)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm.tqdm(executor.map(process_image, imgs_path), total=len(imgs_path)))

if __name__ == '__main__':
    main()
