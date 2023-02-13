"""
util script that generates all the dataset pickled files
(instead of generating them at runtime)
"""

from tqdm import tqdm
import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..')

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from datasets.simulation_ds import SimulationFastDataset

FRAMERATES = [1,2,4,10,20]
SVS_INIT = [(1,3,5), (2,3,10), (4,1,5)]
TRIGGERING = [True, False]
SIMULATOR = ['static', 'grey']
tot = len(FRAMERATES)*len(SVS_INIT)*len(TRIGGERING)*len(SIMULATOR)

args = get_args_parser().parse_args()
args.dont_cache = False
args.use_cars=True
args.crop_svs=True
with tqdm(total=tot) as pbar:
    for c,o,h in SVS_INIT:
        args.svs_close = c
        args.svs_open  = o
        args.svs_hot   = h
        for fr in FRAMERATES:
            args.framerate = fr
            for tr in TRIGGERING:
                args.triggering = tr
                for sim in SIMULATOR:
                    args.simulator = sim
                    try:
                        dataset = FastDataset(args, False, False)
                        dataset = FastDataset(args, True, False)
                        dataset = SimulationFastDataset(args)
                    except Exception as e: print(e)
                    pbar.update(1)