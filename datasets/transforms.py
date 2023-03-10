import copy
import random
import numpy as np
import cv2

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T



def gaussian_noise(img, mask, sigma):
    # make it a tensor
    img = F.to_tensor(img)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    # add noise
    noise = F.gaussian_blur(sigma * torch.randn_like(img), 5)
    noise = noise * mask
    img = (img+noise).clamp(0,1)

    # convert to PIL
    if img.dtype != dtype:
        img = img.to(dtype)
    img = F.to_pil_image(img)
        
    return img

def get_mask(img, position):
    if position is None: return None, None
    # mask
    img = F.to_tensor(img).mean(dim=0)
    y,x = int(position[0]*img.shape[0]), int(position[1]*img.shape[1])
    color = img[y,x]
    mask1 = (img-color)**2 < 0.05**2
    h,w = int(position[2]*img.shape[0]), int(position[2]*img.shape[1])
    mask2 = torch.zeros_like(mask1)
    mask2[max(0,y-h):y+h, max(0,x-w):x+w] = True
    # sigma
    mask = mask1 & mask2
    sigma = 5. + 30*(1- mask.sum()/mask.numel())*position[3]

    return mask[None], sigma.item() / 255

def augment_color(img, brightness, contrast, saturation, sharpness, hue, gamma, gnoise_mask=None, gnoise_sigma=None, **kw):
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

    if gnoise_mask is not None:
        img = gaussian_noise(img, gnoise_mask, gnoise_sigma)

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
    if len(im.shape)==2:
        im = im[:,:,None]

    # Update Targets
    tg *= r
    tg[:, 0] += left
    tg[:, 1] += top

    # keep targets with height > 10 ############ IMPORTANT
    tg = tg[tg[:,3]>10]

    return im, tg



def gettransforms_post(aug_post, out_shape):
    if aug_post:
        motaug = MotCompose([
                    # rotation ?
                    MotRandomShift(*out_shape,False), # shift/resize
                    MotRandomHorizontalFlip(),
                    MotToTensor(),  # also scales from HW to [01]
                    MotNormalize([0.1], [0.9])
                ])
    else:
        motaug = MotCompose([
                    MotRandomShift(*out_shape,True),
                    MotToTensor(),  # also scales from HW to [01]
                    MotNormalize([0.1], [0.9])
                ])

    def fn(img, boxes, ids):
        img = F.to_pil_image((torch.tensor(img)/255).permute(2,0,1))
        boxes = torch.tensor(boxes).view(-1,4)
        boxes[:, 2:] += boxes[:, :2] # boxes from x1y1wh to x1y1x2y2 
        imgs,tgs = motaug([img], [{'boxes':boxes, 'obj_ids':torch.tensor(ids)}]) # boxes from x1y1x2y2 to xcycwh
        return imgs[0], tgs[0]['boxes'], tgs[0]['obj_ids']
    return fn

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class MotRandomShift():
    def __init__(self, h: int, w: int, test: bool):
        self.h = h
        self.w = w
        self.test = test

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []
        w, h = self.w, self.h

        # deterministic or random
        r = 0.9 if self.test else np.random.rand()

        # image size is already 128,160 or bigger
        if w == imgs[0].size[0]:
            r = r*.2+.8     # 80/90 % of the image
        else:
            if self.test: r = 0.7
            r = 1+r*2       # 1/3-100 % of the image  (args.use_crop)
        
        # deterministic or random
        if self.test:
            i,j = imgs[0].size[1]-int(r*h), imgs[0].size[0]-int(r*w)
            region = i//2, j//2, int(r*h),int(r*w)
        else: 
            region = T.RandomCrop.get_params(imgs[0], [int(r*h),int(r*w)])

        for img, target in zip(imgs, targets):
            img_i, target_i = random_shift(img, target, region, (h, w))
            ret_imgs.append(img_i)
            ret_targets.append(target_i)

        return ret_imgs, ret_targets
"""
class MotCropIfBigger(object):
    def __init__(self, h: int, w: int, test: bool):
        self.h = h
        self.w = w
        self.test = test

    def __call__(self, imgs: list, targets: list):
        ret_imgs = []
        ret_targets = []

        if self.test:   # fixed crop
            i,j = imgs[0].size[1]-self.h-36, imgs[0].size[0]-self.w-100
            region = i, j, self.h, self.w
        else:           # rand crop
            region = T.RandomCrop.get_params(imgs[0], [self.h, self.w])

        for img_i, targets_i in zip(imgs, targets):
            img_i, targets_i = crop_mot(img_i, targets_i, region)
            ret_imgs.append(img_i)
            ret_targets.append(targets_i)
        return ret_imgs, ret_targets

def crop_mot(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    boxes = target["boxes"]
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    target["boxes"] = cropped_boxes.reshape(-1, 4)


    cropped_boxes = target['boxes'].reshape(-1, 2, 2)
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

    target['boxes'] = target['boxes'][keep]
    target['obj_ids'] = target['obj_ids'][keep]

    return cropped_image, target"""

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



def random_shift(image, target, region, sizes):
    oh, ow = sizes
    # step 1, shift crop and re-scale image firstly
    cropped_image = F.crop(image, *region)
    cropped_image = F.resize(cropped_image, sizes, interpolation=F.InterpolationMode.NEAREST)

    target = target.copy()
    i, j, h, w = region

    # update boxes
    cropped_boxes = target["boxes"] - torch.as_tensor([j, i, j, i])
    cropped_boxes *= torch.as_tensor([ow / w, oh / h, ow / w, oh / h])
    cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
    max_size = torch.as_tensor([ow-1, oh-1], dtype=torch.float32)
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    target["boxes"] = cropped_boxes.view(-1,4)

    # # select by in screen
    keep1 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

    # select by cut
    hwratio = torch.tensor(list(map(lambda box:  (box[3]-box[1])/(box[2]-box[0]+1e-4), cropped_boxes.view(-1, 4))))
    keep2 = hwratio < 8

    # select by area #NOTE: kill bounding boxes less 8 pixels
    areas = torch.tensor(list(map(lambda box:  (box[3]-box[1])*(box[2]-box[0]), cropped_boxes.view(-1, 4))))
    keep3 = areas > 7

    keep = keep1 & keep2 & keep3
    target['boxes'] = target['boxes'][keep]
    target['obj_ids'] = target['obj_ids'][keep]

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
