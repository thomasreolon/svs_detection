from tqdm import tqdm
import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..')

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset

FRAMERATES = [1,4,16]
SVS_INIT = [(1,3,5), (2,3,10), (4,1,5)]
USE_CROP = [True, False]
SIMULATOR = ['static', 'grey']
tot = len(FRAMERATES)*len(SVS_INIT)*len(USE_CROP)*len(SIMULATOR)*2


args = get_args_parser().parse_args()
args.dont_cache = False
with tqdm(total=tot) as pbar:
    for c,o,h in SVS_INIT:
        args.svs_close = c
        args.svs_open  = o
        args.svs_hot   = h
        for fr in FRAMERATES:
            args.framerate = fr
            for crop in USE_CROP:
                args.use_crop = crop
                for sim in SIMULATOR:
                    args.simulator = sim
                    pbar.update(1)
                    dataset = FastDataset(args, False, False)
                    pbar.update(1)
                    dataset = FastDataset(args, True, False)