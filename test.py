import os
import cv2

dirname  = os.path.dirname(__file__)
input_dir = os.path.join(dirname, 'images', 'input')
raw_dir = os.path.join(input_dir, 'selected', 'raw')
cropped_dir = os.path.join(input_dir, 'selected', 'cropped')

def load_images(path) -> {}:
    result = {}
    imgs = os.listdir(path)
    for img in imgs:
        result[img.replace('.png', '')] = cv2.imread(os.path.join(path, img))
    return result



raw = load_images(raw_dir)
cropped = load_images(cropped_dir)

cv2.