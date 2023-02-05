import copy
import random
import PIL
import numpy as np
import cv2

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.ops.misc import interpolate



def augment_color(img, brightness, contrast, saturation, sharpness, hue, gamma):
    """
    Args
        img: PIL Image
        others: change factor, 0.5 means the same    (from 0.43 to 0.57 the change is skipped to improve speed)
    """
    vars =   [brightness, contrast,   saturation, sharpness,   hue,        gamma      ]
    center = [(0.14, 1.1), (0.8, 0.5), (0.9, 0.22), (0.9, 0.15), (-.1, 0.2), (1.25, -0.5)]
    fns =    [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_sharpness, F.adjust_hue, F.adjust_gamma]

    for var, fn, (base, r) in zip(vars, fns, center):
        if (var-0.5)**2 > 0.005:
            var = min(1,max(0,var))
            factor = base + var*r
            img = fn(img, factor)

    return img

def gettransform_numpygrayscale():
    transform = T.Compose([T.Grayscale(), T.ToTensor()])
    def fn(img):
        t_img = transform(img)
        n_img = (t_img*255).permute(1,2,0).numpy().astype(np.uint8)
        return n_img
    return fn

def letterbox(im, tg, new_shape=(128, 160)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=66)  # add darkgray(66) border

    # Update Targets
    tg *= r
    tg[:, 0] += left
    tg[:, 1] += top

    # keep targets with height > 10 ############ IMPORTANT
    tg = tg[tg[:,3]>10]

    return im[:,:,None], tg



def gettransforms_motaugment():
    motaug = MotCompose([
                MotRandomHorizontalFlip(),
                FixedMotRandomShift(),
                MotToTensor(),  # also scales from HW to [01]
                MotNormalize([0.1, 0.1, 0.1], [0.9, 0.9, 0.9])
            ])
    def fn(img, boxes, ids):
        imgs,tgs = motaug([img], [{'boxes':boxes, 'obj_ids':ids}])
        return imgs[0], tgs[0]['boxes'], tgs[0]['obj_ids']
    return fn

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class MotRandomHorizontalFlip():
    def __call__(self, imgs, targets):
        if random.random() < 0.5:
            ret_imgs = []
            ret_targets = []
            for img_i, targets_i in zip(imgs, targets):
                img_i, targets_i = hflip(img_i, targets_i)
                ret_imgs.append(img_i)
                ret_targets.append(targets_i)
            return ret_imgs, ret_targets
        return imgs, targets

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    boxes = target["boxes"]
    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
    target["boxes"] = boxes

    return flipped_image, target


class FixedMotRandomShift():
    def __init__(self, bs=1, padding=64):
        self.bs = bs
        self.padding = padding

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []

        n_frames = self.bs
        w, h = imgs[0].size
        xshift = (self.padding * torch.rand(self.bs)).int() + 1
        xshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        yshift = (self.padding * torch.rand(self.bs)).int() + 1
        yshift *= (torch.randn(self.bs) > 0.0).int() * 2 - 1
        ret_imgs.append(imgs[0])
        ret_targets.append(targets[0])
        for i in range(1, n_frames):
            ymin = max(0, -yshift[0])
            ymax = min(h, h - yshift[0])
            xmin = max(0, -xshift[0])
            xmax = min(w, w - xshift[0])
            prev_img = ret_imgs[i-1].copy()
            prev_target = copy.deepcopy(ret_targets[i-1])
            region = (int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin))
            img_i, target_i = random_shift(prev_img, prev_target, region, (h, w))
            ret_imgs.append(img_i)
            ret_targets.append(target_i)

        return ret_imgs, ret_targets

def random_shift(image, target, region, sizes):
    oh, ow = sizes
    # step 1, shift crop and re-scale image firstly
    cropped_image = F.crop(image, *region)
    cropped_image = F.resize(cropped_image, sizes)

    target = target.copy()
    i, j, h, w = region

    boxes = target["boxes"]
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    cropped_boxes *= torch.as_tensor([ow / w, oh / h, ow / w, oh / h])
    target["boxes"] = cropped_boxes.reshape(-1, 4)

    cropped_boxes = target['boxes'].reshape(-1, 2, 2)
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

    n_size = len(target['obj_ids'])
    target['boxes'] = target['boxes'][keep[:n_size]]
    target['obj_ids'] = target['obj_ids'][keep[:n_size]]

    return cropped_image, target


class MotToTensor():
    def __call__(self, imgs, targets):
        ret_imgs = []
        for img in imgs:
            ret_imgs.append(F.to_tensor(img))
        return ret_imgs, targets


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]

        boxes = target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["boxes"] = boxes
        return image, target

class MotNormalize(Normalize):
    def __call__(self, imgs, targets=None):
        ret_imgs = []
        ret_targets = []
        for i in range(len(imgs)):
            img_i = imgs[i]
            targets_i = targets[i] if targets is not None else None
            img_i, targets_i = super().__call__(img_i, targets_i)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets

class MotCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, targets):
        for t in self.transforms:
            imgs, targets = t(imgs, targets)
        return imgs, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
