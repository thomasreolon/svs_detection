

import torch
from .mot_dataset_svs import MOTDataset




class JointStaticMOT(torch.utils.data.Dataset):
    """Wrapper To Load Multiple Dataset Faster"""

    def __init__(self, 
                 args,              # NameSpace
                 is_train,          # train or test folder
                 sim_params,        # [(close,open,hot), (close2,open2,hot2), ..]
                 framerates,        # [10, 15, ..]
                 ):
        super().__init__()

        videos = ...
        aug = [{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5}] +( [{'brightness':0.1, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2}] if is_train else [{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5}])

        keys = get_combinations(videos, sim_params, framerates)


        for col in [{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5}, {'brightness':0.1, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2}]:

                    key1 = f'{cl}-{op}-{ho}'
                    key2 = f'{vid}-{fr}-{False}-True-{list(col.values())}'





def get_combinations():
    pass



def try_load_from_cache(da_cache_path, )





