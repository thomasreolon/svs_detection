import pickle
import os

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils.visualize import visualize_prediction, load_original

args = get_args_parser().parse_args()



dataset = FastDataset(args, False, False)

for i in range(1):
    info, img, boxes, ids = dataset[i]

    info = info.replace('_00', ';00')
    info = info.replace('_{', ';{')
    info = info.split(';')
    info = info[:2] + ['False'] + info[2:]
    info = ';'.join(info)

    load_original(args, info)





