import numpy as np
import torchvision
from torch.utils.data import DataLoader
import cv2, torch

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from models._head import xyxy2xywh


def main(args):
    # dataset
    loader = DataLoader(FastDataset(args, True, True), 100, True, collate_fn=FastDataset.collate_fn)

    # test
    model = BlobDetector(True)

    _,x,y,_ = next(iter(loader))
    model(x, y)

class BlobDetector():
    def __init__(self, vis=False):
        self.kernel = np.array(kernel).astype(float)
        self.vis = vis

    def __call__(self, x, y=None):
        x = (x>0).int().numpy() # binarize
        results = []
        for i, img in enumerate(x[:,0]):
            boxes = self.forward_one(img) # detect
            boxes = self.nms(boxes)           # non max suppression
            if self.vis: 
                gt = y[y[:,0]==i, 2:] if y is not None else None
                self.show(img, boxes, gt)
            results.append(boxes)
        return results

    def get_points(self, img):
        _, _, boxes, _ = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
        big = boxes[:,-1]>30 ; big[0]=False
        return boxes[big, :4]

    def nms(self, boxes):
        boxes = torch.tensor(boxes).view(-1,4).float()
        i = torchvision.ops.nms(boxes, torch.ones(len(boxes)), 0.3)
        boxes = xyxy2xywh(boxes[i]).numpy().astype(int)
        return boxes

    def show(self, img, detections, gt=None):
        tmp = (img*255).astype(np.uint8)
        tmp = np.stack((tmp,tmp,tmp), axis=2)
        if gt is not None:
            for xc,yc,w,h in gt:
                x1,x2 = int((xc-w/2)*160), int((xc+w/2)*160)
                y1,y2 = int((yc-h/2)*128), int((yc+h/2)*128)
                tmp = cv2.rectangle(tmp, (x1,y1), (x2,y2), (0,222,0),1) # green truth
        for xc,yc,w,h in detections:
            x1,x2 = int((xc-w/2)), int((xc+w/2))
            y1,y2 = int((yc-h/2)), int((yc+h/2))
            tmp = cv2.rectangle(tmp, (x1,y1), (x2,y2), (0,100,222),1) # green truth
        cv2.imshow('img', tmp)
        cv2.waitKey()


    def forward_one(self, img):
        detections = []
        tmp = (img*255).astype(np.uint8)
        for min_score in [0.6,0.3,0.15]:
            boxes = self.get_points(tmp)
            for x,y,w,h in boxes:
                # white pixels of object
                object = img[y+1:y+h-1, x+1:x+w-1]
                if min(*object.shape) == 0: continue

                # cv2.imshow('a', tmp[y+1:y+h-1, x+1:x+w-1])
                # cv2.waitKey()
                p_w = object.sum()
                tot_pixels = (object.shape[0]*object.shape[1])

                if p_w/tot_pixels>min_score:
                    # black pixels around object
                    h1=max(0,y-2);w1=max(0,x-2)
                    black = img[h1:y+h+2, w1:x+w+2]
                    tot_pixels_border = (black.shape[0]*black.shape[1]) - tot_pixels
                    p_b = tot_pixels_border-(black.sum()-p_w)
                    if p_b / tot_pixels_border > 0.8 and h>w:
                        detections.append([x,y,x+w,y+h])
            tmp = cv2.dilate(tmp, np.ones((3,3)))
        return detections

kernel = [
    [-1,-1,-1,1,1,-1,-1,-1],
    [-1,-1,-1,1,1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,1,1,1,1,-1,-1],
    [-1,-1,1,1,1,1,-1,-1],
    [-1,-1,1,1,1,1,-1,-1],
    [-1,-1,1,1,1,1,-1,-1],
    [-1,-1,1,1,1,1,-1,-1],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
]

if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.crop_svs = True
    args.dataset = 'MOT'
    main(args)