from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from sklearn.cluster import KMeans

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
import utils.debugging as D

import cv2

SIZES = [1,4,8]
KERNELS = 8

def main(args):
    kernels = get_clusters(args, len(SIZES)*KERNELS)
    alphas = [0] * len(kernels)
    loader = DataLoader(FastDataset(args, True, True), 100, True)


def get_clusters(args, n):
    # get ideal clusters BB sizes
    dataset = FastDataset(args, True, False)
    boxes = []
    for i in tqdm(range(len(dataset))):
        _,_,y,_ = dataset[i]
        for box in y:
            boxes.append([(box[-2]*160).item(), (box[-1]*128).item()])

    boxes = np.array(boxes)
    clusters = KMeans(n_clusters=n, random_state=1, n_init="auto").fit(boxes)
    anchors = clusters.cluster_centers_.copy()
    anchors = np.array(anchors).round().astype(int)
    kernels = []
    for w,h in anchors:
        empty = torch.zeros(h,w)
        kernels.append([empty, 0])


    # build kernels
    dataset = FastDataset(args, True, True)
    for i in tqdm(range(len(dataset))):
        _,img,y,_ = dataset[i]
        img = (img>0).float()
        for box in y:
            # get cluster id
            b = [(box[-2]*160).item(), (box[-1]*128).item()]
            b = clusters.predict([b])[0]


            # get bounding box
            xc,yc,w,h = (box[2:] * torch.tensor([160,128,160,128])).int().tolist()
            x1,y1,x2,y2 = xc-w//2, yc-h//2, xc+w//2, yc+h//2
            crop = img[0,y1:y2,x1:x2]
            if crop.numel()==0:print('skip');continue
            ch, cw = crop.shape

            # update kernel
            kernels[b][1] += 1
            matrix = kernels[b][0]
            mh, mw = matrix.shape

            if mh > ch:
                y1c,y2c = 0, ch
                t = (mh-ch) //2
                y1m,y2m = t, ch+t
            else:
                y1m,y2m = 0, mh
                t = (ch-mh) //2
                y1c,y2c = t, mh+t

            if mw > cw:
                x1c,x2c = 0, cw
                t = (mw-cw) //2
                x1m,x2m = t, cw+t
            else:
                x1m,x2m = 0, mw
                t = (cw-mw) //2
                x1c,x2c = t, mw+t

            matrix[y1m:y2m, x1m:x2m] += crop[y1c:y2c, x1c:x2c]

    kern = []
    for i, (k, n) in enumerate(kernels):
        # get average shape
        k = k - n//5
        k = (k>0).float()
        kern.append(k)
    kern = sorted(kern, key=lambda k:k.numel())
    kernels = [[] for _ in range(len(SIZES))]
    for i, k in enumerate(kern):
        down = SIZES[i//KERNELS]
        k = cv2.resize(k.numpy(), (k.shape[1]//down, k.shape[0]//down), interpolation=cv2.INTER_NEAREST)
        kernels[i//KERNELS].append(k)
        cv2.imshow(f'{i}', k)
    cv2.waitKey()
    return kernels



if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.crop_svs = True
    main(args)