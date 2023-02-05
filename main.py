import pickle
import os

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset

args = get_args_parser().parse_args()



dataset = FastDataset(args, True, False)

# for i in range(10):
#     info, img, boxes, ids = dataset[i]





