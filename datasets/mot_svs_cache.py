import os
import numpy as np
import torch
from tqdm import tqdm

from .mot_dataset_svs import MOTDataset

"""
Wrapper around mot_dataset_svs.MOTDataset
--> defines groups of videos (in get_configs())
--> returns a FastDataset that merges the group of videos for that configuration
"""

class FastDataset(torch.utils.data.Dataset):
    """Wrapper To Load Multiple Dataset Faster"""

    def __init__(self, 
                 args,              # NameSpace
                 is_train,          # train or test folder
                 aug_affine,        # augment post svs
                 ):
        super().__init__()
        dataset_configs = get_configs(args, is_train, aug_affine, args.dataset)
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
        "drops dataset with key similar to to ones specified in names"
        keys = list(self.datasets.keys())
        for n in names:
            for k in keys:
                if n in k and k in self.datasets:
                    del self.datasets[k]

    @staticmethod
    def collate_fn(batch):
        return MOTDataset.collate_fn(batch)

def get_cache_path(args, v):
    if args.dont_cache: return None
    os.makedirs( f'{args.out_path}/_ds_cache/', exist_ok=True)
    noise = 'None' if v["aug_color"]["noise"] is None else ",".join([f'{x:.2f}' for x in v["aug_color"]["noise"]])
    selection = ','.join(sorted(v["select_video"])) if isinstance(v["select_video"], list) else v["select_video"]
    triggering = '_TRG' if v["triggering"] else ''
    tmp = v["policy"].replace("\\","/").split("/")[-1] #fname
    policy = f'_POL[{tmp.split(".")[0]}]' if v["simulator"]=='policy' else ''
    ker = f',{v["svs_ker"]}' if v["svs_ker"]!=0 else ''
    svs = f'_SVS[{v["svs_close"]},{v["svs_open"]},{v["svs_hot"]}{ker}]' if v["simulator"]!='grey' else ''
    return  f'{args.out_path}/_ds_cache/ds' \
            f'{svs}' \
            f'_SEL[{selection}]' \
            f'_FRM[{v["framerate"]}]' \
            f'_CAR[{int(v["use_cars"])}]' \
            f'_TRN[{int(v["is_train"])}]' \
            f'_CRP[{int(v["crop_svs"])}]' \
            f'{policy}' \
            f'{triggering}' \
            f'_SIM[{v["simulator"]}]' \
            f'_COL[{v["aug_color"]["brightness"]:.2f},{v["aug_color"]["contrast"]:.2f},' \
                 f'{v["aug_color"]["saturation"]:.2f},{v["aug_color"]["sharpness"]:.2f},' \
                 f'{v["aug_color"]["hue"]:.2f},{v["aug_color"]["gamma"]:.2f},{noise}]' \
             '.pkl'

def get_configs(args, is_train, aug_affine, dataset):
    st0 = np.random.get_state()
    np.random.seed(seed=42)
    configs = {}
    if is_train:
        if dataset=='prova':
            configs.update({
                'prova':{
                    # all videos from MOT17 without augmentations
                    'select_video':'synth-513',
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':False,   
                }})

        if dataset=='all' or dataset=='people' or dataset == 'MOT':
            configs.update({
                'mot17':{
                    # all videos from MOT17 without augmentations
                    'select_video':'MOT17',
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':False,   
                },
                'synth':{
                    # all videos from synthMOT without augmentations
                    'select_video':'synth',
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':False,   
                },
                'synth-dark':{
                    # some videos from synthMOT with darkness
                    'select_video':'synth-7',
                    'is_train':True,  
                    'aug_color':{'brightness':0.3, 'contrast':0.3, 'saturation':0.4, 'sharpness':0.5, 'hue':0.6, 'gamma':0.2, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':False,   
                },})
        if dataset=='all' or dataset=='people' or dataset == 'streets23':
            configs.update({
                'streets23':{
                    'select_video':'vid_',
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':args.triggering,   
                },
                'streets23-dark':{
                    'select_video':['vid_2','vid_4'],
                    'is_train':True,  
                    'aug_color':{'brightness':0.2, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':args.triggering,   
                },
                'streets23-noise':{
                    'select_video':['vid_4','vid_2'],
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':(.2,.7,.2,.7)}, 
                    'aug_affine':aug_affine,
                    'triggering':args.triggering,   
                },
            })
        if dataset=='all' or dataset == 'cars':
            configs.update({
                'cars':{
                    'select_video':['crossing'],
                    'is_train':True,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':aug_affine,
                    'triggering':True,   
                }})

    else:
        # test Datasets
        if dataset=='all' or dataset=='people' or dataset == 'MOT':
            configs.update({
                'mot17':{
                    'select_video':['MOT17'],
                    'is_train':False,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':False,
                    'triggering':False,    
                },
                'synth':{
                    'select_video':['synth'],
                    'is_train':False,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':False,
                    'triggering':False,    
                },
                'synth-darknoise':{
                    'select_video':'synth',
                    'is_train':False,  
                    'aug_color':{'brightness':0.2, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.4, 'hue':0.5, 'gamma':0.1, 'noise':(.2,.7,.2,.6)}, 
                    'aug_affine':False,
                    'triggering':False,    
                }
            })
        if dataset=='all' or dataset=='people' or dataset == 'streets23':
            configs.update({
                'streets23':{
                    'select_video':'vid_',
                    'is_train':False,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':False,
                    'triggering':args.triggering,    
                },
                'streets23-darknoise':{
                    'select_video':'vid_',
                    'is_train':False,  
                    'aug_color':{'brightness':0.2, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2, 'noise':(.2,.7,.2,.6)}, 
                    'aug_affine':False,
                    'triggering':args.triggering,    
                }
            })
        if dataset=='all' or dataset == 'cars':
            configs.update({
                'cars':{
                    'select_video':['crossing'],
                    'is_train':False,  
                    'aug_color':{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5, 'noise':None}, 
                    'aug_affine':False,
                    'triggering':True,   
                }})
        
    # General Settings
    for v in configs.values():
        v.update({
            'mot_path':args.mot_path,
            'svs_close':args.svs_close, 
            'svs_open':args.svs_open,  
            'svs_hot':args.svs_hot,   
            'svs_ker':args.svs_ker,   
            'framerate':args.framerate, 
            'use_cars':True,
            'simulator':args.simulator,
            'crop_svs':args.crop_svs,
            'policy': args.policy,
        })
        np.random.set_state(st0)

    return configs
         






