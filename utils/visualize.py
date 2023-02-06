import os
from PIL import Image
import numpy as np
import cv2
import json
import time
from collections import defaultdict

import torchvision.transforms.functional as F
import torch

from datasets.transforms import augment_color, get_mask, letterbox

class StatsLogger():
    VIS_SIZE = (640, 800)
    def __init__(self, args) -> None:
        self.args = args
        self.out_path = f'{args.out_path}/{args.exp_name}/main_logs/'
        os.makedirs(self.out_path, exist_ok=True)
        self.start_time = time.time()
        self.stats = defaultdict(lambda: [np.zeros((2,2)), []])

        self.logtxt, self.logtxt_name = None, self.out_path+f'logs.txt'
        self.cap = None
        assert self.VIS_SIZE[0]/self.VIS_SIZE[1]==128/160

    def log(self, text):
        "write log on log file"
        if self.logtxt is None:
            self.logtxt = open(self.logtxt_name, 'w')
        self.logtxt.write(text)
        self.logtxt.flush()

    def log_time(self):
        if self.logtxt is None:
            self.logtxt = open(self.logtxt_name, 'w')
        tot_s = time.time() - self.start_time
        h = tot_s//3600 ; m = (tot_s%3600)//60 ; s = tot_s%60
        text = f'>> Total Time: {h}h {m}m {s}s\n'
        self.logtxt.write(text)
        self.logtxt.flush()

    def visualize(self, info, svs_img, gt_boxes, gt_ids, pred_boxes, heat, count):
        """generates visualization with (ground truth) and (predictions)
        
        Args
            info    : string with infos about the simulated frame (as returned from dataset)
            svs_img : input of NN, the normalized simulated frame (as returned from dataset)
            gt_boxes: ground truth about objects location (target from dataset, last 4 fields)
            gt_ids  : ground truth about objects identities (as returned from dataset)
            pred_boxes  : NN prediction about objects location, after NMS
            heat    : NN confidence logits
        """
        # load original image and draw GT on it
        ori_img = self.load_original(self.args, info)
        ori_img = self.draw_boxes(ori_img, gt_boxes, gt_ids)[:,:,::-1] # to bgr

        # rescale SVS
        svs_img, pred_boxes = self.clean_svs(svs_img, pred_boxes)
        svs_img = self.draw_boxes(svs_img, pred_boxes, [1]*len(pred_boxes))

        # rescale heatmap
        heat = np.uint8((heat*255)[...,None].expand(-1,-1,3))
        heat = cv2.resize(heat, (self.VIS_SIZE[1], self.VIS_SIZE[0]))
        heat[heat[...,0]>self.args.detect_thresh*255] = (0,0,200)

        # print frame infos
        boxes = np.ones_like(heat)*255  # TODO: view what?
        boxes = self.print_boxes(boxes, gt_boxes, pred_boxes, count)        

        # write on video
        row1 = np.concatenate((ori_img, svs_img), axis=1)
        row2 = np.concatenate((heat, boxes), axis=1)
        visualization = np.concatenate((row1, row2), axis=0)

        self.cap.write(visualization)

    def save_cfg(self):
        "write experiment configs on file"
        args = vars(self.args)
        with open(f'{self.out_path[:-10]}cfg.json', 'w') as fout:
            json.dump(args, fout, indent=2)
    

    def close(self):
        if self.logtxt is not None:
            self.logtxt.close()
        if self.cap is not None:
            self.cap.release()

    def new_video(self, v_name):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoWriter(self.out_path+f'pred_{v_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.args.framerate, (self.VIS_SIZE[1]*2, self.VIS_SIZE[0]*2))

    def print_boxes(self, img, gt_boxes, pred_boxes, count):
        """write boxes"""
        img = cv2.putText(img, 'Annotations',   (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        img = cv2.putText(img, 'Predictions',   (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        i = 2
        for box in gt_boxes:
            img = cv2.putText(img, f'{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}',   (8,i*28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            i += 1
        i = 2
        for box in pred_boxes[:8]:
            img = cv2.putText(img, f'{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}',   (400,i*28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            i += 1
        gt_p, gt_c = int(len(gt_boxes)>0), len(gt_boxes)
        pr_p, pr_c = int(count[0]>.5), int(count[1])
        img = cv2.putText(img, f'Activate: [{gt_p: 2d} ]  [{pr_p: 2d} ]',   (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        img = cv2.putText(img, f'Count:   [{gt_c: 2d} ]  [{pr_c: 2d} ]',   (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        return img

    def draw_boxes(self, img, boxes, ids):
        """draw boxes & ids on frame"""
        boxes = safe_box_cxcywh_to_xyxy(boxes)
        boxes = (boxes * torch.tensor([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]])).int().tolist()
        for (x1,y1,x2,y2), id in zip(boxes, ids):
            if isinstance(id, torch.Tensor): id = id.item()
            color = tuple([(id * prime + (10+id)*83) % 255 for prime in [643,997,676]])
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color)
            img = cv2.putText(img, f"{id}", (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        return img

    def clean_svs(self, svs_img, pred_boxes):
        """posrprocess after YoloDetectionHead for visualization"""
        # normalize precictons (to be in range 0,1)
        pred_boxes = pred_boxes / torch.tensor(svs_img.shape)[[2,1,2,1]]

        # upscale image for better visualization
        svs_img = ((svs_img*.9+.1).permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
        hwc = *self.VIS_SIZE, 3
        svs_img = cv2.resize(svs_img, (hwc[1], hwc[0]))[:,:,None]
        svs_img = np.uint8(np.broadcast_to(svs_img, hwc))[:].copy()

        return svs_img, pred_boxes

    def load_original(self, args, info):
        """loads frame and applies color data augmentation for visualization of ground truth"""
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

    def collect_stats(self, vid, count, pred, boxes):
        cm, c_err = self.stats[vid]
        r,p = int(len(boxes)>0), int(count[0]>.5)
        cm[p,r] += 1
        c_err.append((count[1]-len(boxes)).abs())

    def log_stats(self):
        self.log('_____________STATS____________\n')
        cum = []
        for vid, (cm, c_err) in self.stats.items():
            acc = cm.diagonal().sum() / cm.sum()
            pr = cm[1,1] / (cm[:,1].sum()+1e-8)
            rc = cm[1,1] / (cm[1,:].sum()+1e-8)
            c_err = np.array(c_err)
            em, es = c_err.mean(), c_err.std()
            text = f'[{vid}] Detection[accuracy={acc:.2f}  precision={pr:.2f}  recall={rc:.2f}]  Count[meanerr={em:.2f}  std={es:.1e}]\n'
            self.log(text)
            if 'test' in vid:
                cum.append([acc,pr,rc])
        acc,pr,rc = np.array(cum).mean(axis=0).tolist()
        text = f'>> Averaged Stats For Testset [accuracy={acc:.2f}  precision={pr:.2f}  recall={rc:.2f}]\n'
        self.log(text)

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