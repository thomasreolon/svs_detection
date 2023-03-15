from tqdm import tqdm
import os
import pickle

from .mot_svs_cache import FastDataset, get_configs, MOTDataset, get_cache_path

"""must use batch size = 1
    Wrapper around mot_svs_cache.FastDataset --> returns greyscale batched by video
"""

class SimulationFastDataset(FastDataset):
    """does not return the whole dataset, just the initial frames for each video"""
    def __init__(self, 
                 args,              # NameSpace
                 num_frames = 100   # how many frames max per video
                 ):
        # like Fast Dataset but Forcing NO_SKIP_FRAMES
        super(FastDataset, self).__init__()

        # load cached datsets
        dataset_configs = get_configs(args, True, False, args.dataset)
        for v in dataset_configs.values():
            v['simulator'] = 'grey'
        self.datasets = {k:MOTDataset(**v, raw=True, cache_path=get_cache_path(args, v)) for k,v in tqdm(dataset_configs.items(),desc='loading dataset pt.1')}

        # divide index by videos
        file_cache = f'{args.out_path}/_ds_cache/_simulatoridx_FR[{args.framerate}].pkl'
        if not os.path.exists(file_cache):
            self.ds_idxs = []
            prev_name, c = None, None
            for i in tqdm(range(super(SimulationFastDataset, self).__len__()),desc='loading dataset pt.2'):
                info, _, _, _ = super(SimulationFastDataset, self).__getitem__(i)
                curr_name = info.split(';')[0] +':'+ info.split(';')[-1]

                if prev_name != curr_name:
                    if i>0:
                        self.ds_idxs.append((c, min(i, c+num_frames)))
                    prev_name, c = curr_name, i

            if not args.dont_cache:
                with open(file_cache, 'wb') as f_data:
                    pickle.dump(self.ds_idxs, f_data)
        else:
            with open(file_cache, 'rb') as f_data:
                self.ds_idxs = pickle.load(f_data)

    def __len__(self):
        return len(self.ds_idxs)

    def __getitem__(self, idx):
        idxes = self.ds_idxs[idx]
        batch = [super(SimulationFastDataset, self).__getitem__(x) for x in range(*idxes)]
        
        tmp = FastDataset.collate_fn(batch)
        return tmp
    
    @staticmethod
    def collate_fn(batch):
        return batch[0]
    
    def drop(self, names):
        raise NotImplementedError()