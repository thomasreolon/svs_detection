from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import StatsLogger, init_seeds, quantize
from models import build as build_model, ComputeLoss
from engine import train_one_epoch, test_epoch
import utils.debugging as D


args = get_args_parser().parse_args()
args.dataset = 'all'

dataset = FastDataset(args, True, False)





