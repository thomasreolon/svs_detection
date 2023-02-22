import os
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
import cv2
import pickle

import torch

from simulators import get_simulator
from .transforms import (
    augment_color,
    get_mask,
    letterbox, 
    gettransform_numpygrayscale, 
    gettransforms_post
)


class MOTDataset(torch.utils.data.Dataset):
    """returns simulated frames of forensor & BBoxes"""
    IMG_SHAPE = 128,160
    NO_AUGCOL = {'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}

    def __init__(self,
                 mot_path,              # path to Dataset
                 svs_close=1,           # param simulator 1
                 svs_open=3,            # param simulator 2
                 svs_hot=5,             # param simulator 3
                 select_video='',       # partial name of video
                 framerate=15,           #  
                 use_cars=False,        # detection of person & cars
                 is_train=True,         # from train or test dataset
                 aug_color=None,        # dict on color augmentation of video
                 aug_affine=True,       # if true use hflip/shift,   else just normalize
                 simulator='static',    # which simulator to use
                 crop_svs=False,        # simulates in high res, will crop later
                 triggering=False,      # if False drops most of frames that do not contain annotations
                 raw = False,           # return frames without pre-proc (does not discard initial ones & empty) 
                 policy = False,        # path to policy weights for policy sensor

                 cache_path=None        # if provided will try to load data from the file insted of simulating it
                 ):
        super().__init__()
        self.aug_post = gettransforms_post(aug_affine, self.IMG_SHAPE)
        self.aug_color = aug_color if aug_color is not None else self.NO_AUGCOL
        self.crop_svs = crop_svs

        # simulator:  frame --> motion_map
        foresensor = get_simulator(simulator, svs_close, svs_open, svs_hot, policy)
        
        if cache_path is not None:
            cache_path = os.path.abspath(cache_path)

        if cache_path is None or not os.path.exists(cache_path):
            # load videos[(img_path, boxes)]
            videos = load_data(mot_path, select_video, framerate, is_train, use_cars)

            # applies aug_color to img ; applies foresensor
            # data[(video_frame, svs_img, boxes, obj_ids)]
            self.data = []
            for video in videos:
                self.data.append(simulate_svs(foresensor, video, self.aug_color, self.IMG_SHAPE, is_train, crop_svs))
            
            if cache_path is not None:
                # save in file_system
                with open(cache_path, 'wb') as f_data:
                    pickle.dump(self.data, f_data)
        else:
            # load from file_system
            with open(cache_path, 'rb') as f_data:
                self.data = pickle.load(f_data)
        
        if raw: 
            # keep all frames
            self.data = sum(self.data, [])
        else:   
            # discard first 1/4 frames (gives time to adapt to simulator)
            self.data = sum([vid[max(2, len(vid)//4):] for vid in self.data], [])
        
        if not (raw or triggering):
            # drops most frames without BB to improve detections
            self.data = [x for x in self.data if self.allow_empty(x[2])]

    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        info, img, boxes, ids = self.data[idx]

        if len(ids) != len(boxes):#NOTE:ids are not useful anyway..
            ids = (list(ids) + list(range(100,200)))[:len(boxes)]

        # augment flip rotate
        img, boxes, ids = self.aug_post(img, boxes, ids)

        # YOLO loss needs 2 additional fields (batch_n, class_label)
        targets = torch.cat((torch.zeros_like(boxes)[:,:2], boxes), dim=1)

        return [info, img, targets, ids]

    @staticmethod
    def collate_fn(batch):
        info, imgs, tgs, ids = zip(*batch)

        for i, tg in enumerate(tgs):
            tg[:, 0] = i
        tgs = torch.cat(tgs, dim=0)     # concat & update batch value

        imgs = torch.stack(imgs)   # concat tensors

        return info, imgs, tgs, ids

    _empty=0
    def allow_empty(self,box):
        if len(box):
            self._empty = min(23,self._empty+1)
            return True
        elif self._empty >= 8:
            self._empty  -= 8
            return True
        return False

def is_in(select, v_name):
    if isinstance(select, str):
        return select in v_name
    else:
        return any([s in v_name for s in select])

def load_data(mot_path, select_video, framerate, is_train, use_cars=False):
    folder = 'train' if is_train else 'test'
    mot_path = f'{mot_path}/{folder}'
    videos = [v_name for v_name in os.listdir(mot_path) if is_in(select_video, v_name)]

    data = []
    for v_name in videos:
        vid_path = f'{mot_path}/{v_name}'
        # get framerate
        with open(f'{vid_path}/seqinfo.ini') as fin:
            fps = int(fin.read().split('frameRate=')[1].split('\n')[0])

        # load images
        img_path = f'{vid_path}/img1/'
        all_imgs = sorted(os.listdir(img_path))

        # load GT
        all_box = defaultdict(lambda: [])
        with open(f'{vid_path}/gt/gt.txt', 'r') as fin:
            for line in fin.readlines():
                if len(line)<10:continue # skip empty lines

                # TODO: faster with np.loadtxt
                if 'synth' in v_name:
                    line = [float(x) for x in line.split(',')]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <confidence> <class> <visibility>  <X><Y><Z>
                    if line[7] != 1 and not use_cars : continue
                    if line[8] < .44: continue
                elif 'MOT17' in v_name:
                    line = [float(x) for x in line.split(',')]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <? iscrowd> <class> <visibility>
                    if line[7] != 1: continue 
                elif 'vid' in v_name:
                    line = [float(x) for x in line.split(' ')]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <-1> <-1> <iscar>
                    if line[8] == 1 and not use_cars: continue # don't get BB of cars if use_cars==False

                all_box[line[0]].append([line[2], line[3], line[4], line[5],  int(line[1])])

        # select by framerate
        selected = []
        step = fps / framerate
        i = 0
        while i < len(all_imgs):
            image_path = img_path+all_imgs[int(i)]
            frame = int(all_imgs[int(i)].split('.')[0]) # int(i) and frame+1 should be ==
            selected.append((image_path, all_box[frame]))
            i += step        
        data.append(selected)
    return data


def simulate_svs(foresensor, data, aug_color, img_shape, is_train, crop_svs):
    # prep images for simulator
    to_gray = gettransform_numpygrayscale()
    images = []
    boxes = []
    obj_id = []
    infos = []
    ff = min(len(data), 120) ; ii = max(0, ff-80) #NOTE: max 80 frames x sequence to speed up
    for i, (im_path, gt) in enumerate(data[ii:ff]):  
        # augment input video color
        img = Image.open(im_path)
        if i==0:
            mask, sigma = get_mask(img, aug_color['noise'])
            aug_color.update({'gnoise_mask':mask, 'gnoise_sigma':sigma})
        img = augment_color(img, **aug_color)
        # img.show('fr')
        img = to_gray(img)

        # resize image for simulator
        gt = np.array(gt).reshape(-1,5)
        if crop_svs and 'vid_' not in im_path:
            crop_svs = False 
            img_shape = (img_shape[0]*3, int((img.shape[1]/img.shape[0]) *img_shape[0]*3))
        img, boxs = letterbox(img, gt[:,:4].astype(np.float64), img_shape)

        images.append(img)
        boxes.append(boxs)
        obj_id.append(gt[:,4])
        tmp = {k:v for k,v in aug_color.items() if k not in {'gnoise_mask', 'gnoise_sigma'}}
        infos.append(f'{im_path.split("/")[-3]};{im_path.split("/")[-1][:6]};{is_train};{str(tmp)}')

    # warm up simulator
    skip = max(1, int(len(images)//20))
    init = np.concatenate(images[::skip], axis=2).mean(axis=2)
    std  = np.concatenate(images[1:10] + images[::skip*3], axis=2).std(axis=2)
    foresensor.init_video(init, std)

    # simulate
    svs_images = [foresensor(img) for img in images]

    # for s,i,b in zip(svs_images, images, boxes):
    #     for box in b:
    #         x1,y1 = int(box[0]), int(box[1])
    #         x2,y2 = int(box[0]+box[2]), int(box[1]+box[3])
    #         i = cv2.rectangle(i, (x1,y1), (x2,y2), (0))
    #     fr = np.concatenate((s,i), axis=1)
    #     cv2.imshow('fr', fr)
    #     cv2.waitKey()
    
    return list(zip(infos, svs_images, boxes, obj_id))


