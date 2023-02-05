import os
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as F
import torch

from datasets.transforms import augment_color, get_mask, letterbox



class StatsLogger():
    VIS_SIZE = (640, 800)
    def __init__(self, args) -> None:
        self.args = args
        self.out_path = args.out_path + '/visualizations/'
        os.makedirs(self.out_path, exist_ok=True)
        self.cap = cv2.VideoWriter(self.out_path+f'{int(torch.rand(1)*3)}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (self.VIS_SIZE[1]*2, self.VIS_SIZE[0]))

    def __call__(self, info, svs_img, gt_boxes, gt_ids, pred_boxes):
        """generates visualization with (ground truth) and (predictions)
        
        Args
            info    : string with infos about the simulated frame (as returned from dataset)
            svs_img : input of NN, the normalized simulated frame (as returned from dataset)
            gt_boxes: ground truth about objects location (target from dataset, last 4 fields)
            gt_ids  : ground truth about objects identities (as returned from dataset)
            pred_boxes  : NN prediction about objects location
        """
        ori_img = self.load_original(self.args, info)
        svs_img = ((svs_img*.9+.1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
        hwc = *self.VIS_SIZE, 3
        svs_img = cv2.resize(svs_img, (hwc[1], hwc[0]))[:,:,None]
        svs_img = np.uint8(np.broadcast_to(svs_img, hwc))[:].copy()

        ori_img = self.draw_boxes(ori_img, gt_boxes, gt_ids)[:,:,::-1] # to bgr

        svs_img = self.draw_boxes(svs_img, pred_boxes, [1]*len(pred_boxes))
        visualization = np.concatenate((ori_img, svs_img), axis=1)

        self.cap.write(visualization)
        cv2.imshow('fr', visualization)
        cv2.waitKey(40)

    def draw_boxes(self, img, boxes, ids):
        """draw boxes & ids on frame"""
        boxes = safe_box_cxcywh_to_xyxy(boxes)
        boxes = (boxes * torch.tensor([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]])).int().cpu().tolist()
        for (x1,y1,x2,y2), id in zip(boxes, ids):
            if isinstance(id, torch.Tensor): id = id.item()
            color = tuple([(id * prime + (10+id)*83) % 255 for prime in [643,997,676]])
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color)
            img = cv2.putText(img, f"{id}", (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        return img

    def load_original(self, args, info):
        """loads frame and applies color data augmentation for visualization of groung truth"""
        info = info.split(';')
        vid = info[0]
        frame = info[1]
        folder = 'train' if eval(info[2]) else 'test'
        info[3] = hack_old_compatibility(info[3])
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
        img, _ = letterbox(img, torch.zeros(0,4), self.VIS_SIZE )
        return img


def safe_box_cxcywh_to_xyxy(x, m_ = 1e-2):
    m_ = torch.tensor([m_],device=x.device,dtype=x.dtype)
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        torch.min(1-m_,  torch.max(m_*0, (x_c - 0.5 * w))), 
        torch.min(1-m_,  torch.max(m_*0, (y_c - 0.5 * h))),
        torch.min(m_/m_, torch.max(m_,   (x_c + 0.5 * w))), 
        torch.min(m_/m_, torch.max(m_,   (y_c + 0.5 * h))),
    ]
    return torch.stack(b, dim=-1)


def hack_old_compatibility(text):
    """__old pickled dataset has a bug __"""
    if 'gnoise_mask' in text:
        text = text.split('\'gnoise_mask')[0]
        text += '}'
    return text