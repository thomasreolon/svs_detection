from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as F

from datasets.transforms import augment_color, get_mask

def visualize_prediction(ori_img, svs_img, gt_bb, pred_bb):
    pass




def load_original(args, info):
    info = info.split(';')
    vid = info[0]
    frame = info[1]
    folder = 'train' if eval(info[2]) else 'test'
    aug_color = eval(info[3])
    frame_path = f'{args.mot_path}/{folder}/{vid}/img1/{frame}.jpg'

    # mask
    tmp = Image.open(f'{args.mot_path}/{folder}/{vid}/img1/000001.jpg')
    mask, sigma = get_mask(tmp, aug_color['noise'])
    aug_color.update({'gnoise_mask':mask, 'gnoise_sigma':sigma})

    # augment color
    img = Image.open(frame_path)
    img = augment_color(img, **aug_color)

    # to numpy
    img = (F.to_tensor(img).permute(1,2,0)*255).numpy().astype(np.uint8)

    return img
