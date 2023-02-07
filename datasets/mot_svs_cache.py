import os
import numpy as np
import torch
from tqdm import tqdm

from .mot_dataset_svs import MOTDataset




class FastDataset(torch.utils.data.Dataset):
    """Wrapper To Load Multiple Dataset Faster"""

    def __init__(self, 
                 args,              # NameSpace
                 is_train,          # train or test folder
                 aug_affine,        # augment post svs
                 ):
        super().__init__()
        dataset_configs = get_configs(args, is_train, aug_affine)
        self.datasets = {k:MOTDataset(**v, cache_path=get_cache_path(args, v)) for k,v in tqdm(dataset_configs.items(),desc='loading dataset')}


    def __len__(self):
        return sum([len(x) for x in self.datasets.values()])


    def __getitem__(self, idx):
        for variant, dataset in self.datasets.items():
            l = len(dataset)
            if idx < l:
                res = dataset[idx]
                # add 'video variant' information to info
                res[0] = res[0]+f';{variant}'
                return res
            idx -= l

    def drop(self, names):
        for n in names:
            if n in self.datasets:
                del self.datasets[n]

    @staticmethod
    def collate_fn(batch):
        return MOTDataset.collate_fn(batch)



def get_cache_path(args, v):
    if args.dont_cache: return None
    os.makedirs( f'{args.out_path}/_ds_cache/', exist_ok=True)
    noise = 'None' if v["aug_color"]["noise"] is None else ",".join([str(x) for x in v["aug_color"]["noise"]])
    simulator = '' if v["simulator"] == 'static' else f'_SIM[{v["simulator"]}]'
    return  f'{args.out_path}/_ds_cache/ds' \
            f'_SVS[{v["svs_close"]},{v["svs_open"]},{v["svs_hot"]}]' \
            f'_SEL[{v["select_video"]}]' \
            f'_FRM[{v["framerate"]}]' \
            f'_CAR[{int(v["use_cars"])}]' \
            f'_TRN[{int(v["is_train"])}]' \
            f'{simulator}' \
            f'_COL[{v["aug_color"]["brightness"]},{v["aug_color"]["contrast"]},' \
                 f'{v["aug_color"]["saturation"]},{v["aug_color"]["sharpness"]},' \
                 f'{v["aug_color"]["hue"]},{v["aug_color"]["gamma"]},{noise}]' \
             '.pkl'


def get_configs(args, is_train, aug_affine):
    np.random.seed(seed=42)
    def rand(): return 0.5 + (-.1+np.random.rand())**2 - (-.1+np.random.rand())**2
    if is_train:
        configs =  {
            'all_videos_MOT':{
                # all videos from MOT17 & synthMOT without augmentations
                'select_video':'-',
                'is_train':True,  
                'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                'aug_affine':aug_affine   
            },
            'all_videos_NEW':{
                # all videos from TOMDataset without augmentations
                'select_video':'vid',
                'is_train':True,  
                'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                'aug_affine':aug_affine   
            },
            'darker':{
                'select_video':'vid_1',
                'is_train':True,  
                'aug_color':{'brightness':0.1, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2, 'noise':None}, 
                'aug_affine':aug_affine   
            },
            'noise':{
                'select_video':'vid_1',
                'is_train':True,  
                'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':(.2,.7,.2,.7)}, 
                'aug_affine':aug_affine   
            },
            'mixed1':{
                'select_video':'vid_5',
                'is_train':True,  
                'aug_color':{'brightness':rand(), 'contrast':rand(), 'saturation':rand(), 'sharpness':rand(), 'hue':rand(), 'gamma':rand(), 'noise':(np.random.rand(),np.random.rand(),.1+np.random.rand(),np.random.rand())}, 
                'aug_affine':aug_affine   
            },
            'mixed2':{
                'select_video':'vid_5',
                'is_train':True,  
                'aug_color':{'brightness':rand(), 'contrast':rand(), 'saturation':rand(), 'sharpness':rand(), 'hue':rand(), 'gamma':rand(), 'noise':(np.random.rand(),np.random.rand(),.1+np.random.rand(),np.random.rand())}, 
                'aug_affine':aug_affine   
            },
            'mixed3':{
                'select_video':'vid_5',
                'is_train':True,  
                'aug_color':{'brightness':rand(), 'contrast':rand(), 'saturation':rand(), 'sharpness':rand(), 'hue':rand(), 'gamma':rand(), 'noise':(np.random.rand(),np.random.rand(),.1+np.random.rand(),np.random.rand())}, 
                'aug_affine':aug_affine   
            },
        }
    else:
        configs =  {
            'all videos':{
                'select_video':'',
                'is_train':False,  
                'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                'aug_affine':False   
            },
            'darker':{
                'select_video':'',
                'is_train':False,  
                'aug_color':{'brightness':0.2, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2, 'noise':None}, 
                'aug_affine':False   
            },
            'noise':{
                'select_video':'',
                'is_train':False,  
                'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':(.2,.7,.2,.6)}, 
                'aug_affine':False   
            },
        }
    for v in configs.values():
        v.update({
            'mot_path':args.mot_path,
            'svs_close':args.svs_close, 
            'svs_open':args.svs_open,  
            'svs_hot':args.svs_hot,   
            'framerate':args.framerate, 
            'use_cars':args.use_cars,
            'simulator':args.simulator,
        })

    return configs
         






