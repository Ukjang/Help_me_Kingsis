import numpy as np
import cv2
from PIL import Image

def center_image(img, src_format='OpenCV', dst_format='OpenCV', IMAGE_SIZE=224):
    if src_format == 'OpenCV':
        h, w = img.shape[:-1]
    else:    # Pillow
        h, w = np.array(img).shape[:-1]

    if h > w:
        width, height = IMAGE_SIZE, (h * IMAGE_SIZE) // w
    else:
        width, height = (w * IMAGE_SIZE) // h, IMAGE_SIZE

    if src_format == 'OpenCV':
        interpolation = cv2.INTER_AREA if h + w > 300 else cv2.INTER_CUBIC
        new_img = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
    else:
        new_img = np.array(img.resize((width, height)))

    diff = abs(width - height) // 2
    if h > w:
        final_img = new_img[diff:diff+IMAGE_SIZE, :]
    else:
        final_img = new_img[:, diff:diff+IMAGE_SIZE]

    return final_img if dst_format == 'OpenCV' else Image.fromarray(final_img)