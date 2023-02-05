import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import cv2

import torch

from .forensor_sim import StaticSVS
from .transforms import augment_color, gettransform_numpygrayscale, letterbox, gettransforms_motaugment







class MOTDataset(torch.utils.data.Dataset):
    """returns simulated frames of forensor & BBoxes"""
    IMG_SHAPE = 128,160
    NO_AUGCOL = {'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5}

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
                 aug_affine=True        # use hflip/rotation
                 ):
        super().__init__()
        self.aug_flip = gettransforms_motaugment() if aug_affine else None
        self.aug_color = aug_color if aug_color is not None else self.NO_AUGCOL


        # simulator:  frame --> motion_map
        foresensor = StaticSVS(svs_close, svs_open, svs_hot, self.IMG_SHAPE)

        # load videos[(img_path, boxes)]
        videos = load_data(mot_path, select_video, framerate, is_train, use_cars)

        # applies aug_color to img ; applies foresensor
        # data[(video_frame, svs_img, boxes, obj_ids)]
        self.data = []
        for video in videos:
            self.data += simulate_svs(foresensor, video, self.aug_color, self.IMG_SHAPE)[2:]

    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        info, img, boxes, ids = self.data[idx]

        # augment flip rotate
        if self.aug_flip is not None:
            img, boxes, ids = self.aug_flip(img, boxes, ids)

        return info, img, boxes, ids

    

def load_data(mot_path, select_video, framerate, is_train, use_cars=False):
    folder = 'train' if is_train else 'test'
    mot_path = f'{mot_path}/{folder}'
    videos = [v_name for v_name in os.listdir(mot_path) if select_video in v_name]

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
                    line = [int(float(x)) for x in line.split(',')[:-1]]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <confidence> <class> <visibility>  <X><Y><Z>
                    if line[7] != 1 and not use_cars : continue
                elif 'MOT17' in v_name:
                    line = [int(float(x)) for x in line.split(',')[:-1]]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <? iscrowd> <class> <visibility>
                    if line[7] > 7: continue 
                    if line[7] != 1 and not use_cars : continue 
                elif 'vid' in v_name:
                    line = [int(float(x)) for x in line.split(' ')[:-1]]
                    # <frame> <track> <leftmost> <topmost> <width> <height> <-1> <-1> <iscar>
                    if line[-1] == 1 and not use_cars: continue # don't get BB of cars if use_cars==False

                all_box[line[0]].append([line[2], line[3], line[4], line[5],  line[1]])

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


def simulate_svs(foresensor, data, aug_color, img_shape):
    # prep images for simulator
    to_gray = gettransform_numpygrayscale()
    images = []
    boxes = []
    obj_id = []
    infos = []
    for im_path, gt in data:
        # augment input video color
        img = Image.open(im_path)
        img = augment_color(img, **aug_color)
        # img.show('fr')
        img = to_gray(img)

        # resize image for simulator
        gt = np.array(gt).reshape(-1,5)
        img, boxs = letterbox(img, gt[:,:4].astype(np.float64), img_shape)

        images.append(img)
        boxes.append(boxs)
        obj_id.append(gt[:,4])
        infos.append(f'{im_path.split("/")[-3]}_{im_path.split("/")[-1][:6]}')
    
    # warm up simulator
    skip = max(1, int(len(images)//20))
    init = np.concatenate(images[::skip], axis=2).mean(axis=2)
    std  = np.concatenate(images[1:10] + images[::skip*3], axis=2).std(axis=2)
    foresensor.init_video(init, std)

    # simulate
    svs_images = [foresensor(img[:,:,0])[:,:,None] for img in images]

    # for s,i,b in zip(svs_images, images, boxes):
    #     for box in b:
    #         x1,y1 = int(box[0]), int(box[1])
    #         x2,y2 = int(box[0]+box[2]), int(box[1]+box[3])
    #         i = cv2.rectangle(i, (x1,y1), (x2,y2), (0))
    #     fr = np.concatenate((s,i), axis=1)
    #     cv2.imshow('fr', fr)
    #     cv2.waitKey()
    
    return list(zip(infos, svs_images, boxes, obj_id))


