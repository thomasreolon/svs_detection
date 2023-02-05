from tqdm import tqdm
import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..')

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset

FRAMERATES = [1,4,16]
SVS_INIT = [(1,3,5), (2,3,10), (4,1,5)]
tot = len(FRAMERATES)*len(SVS_INIT)*2


args = get_args_parser().parse_args()
args.dont_cache = False
with tqdm(total=tot) as pbar:
    for fr in FRAMERATES:
        args.framerate = fr
        for c,o,h in SVS_INIT:
            args.svs_close = c
            args.svs_open  = o
            args.svs_hot   = h

            pbar.update(1)
            dataset = FastDataset(args, False, False)
            pbar.update(1)
            dataset = FastDataset(args, True, False)