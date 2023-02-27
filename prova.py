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

def main(args, device):
    device = 'cpu'
    # Setup Model & Loss
    ch_in = 1 if 'cat' not in args.simulator else 2 # mhicatgrey gives 2 channels images
    model = build_model(args.architecture, ch_in).to(device)
    model = quantize(model, 'binary')

    loss_fn = ComputeLoss(model)
    dataset = FastDataset(args, True, True)

    _,x,gt,_ = dataset[2]
    _,y,c = model(x[None])

    # y[0].sum().backward()
    loss,_ = loss_fn(y,gt,c)
    loss.backward()

if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.crop_svs=True
    if args.architecture=='blob':raise Exception('you should call mainblob.py')
    args.framerate = int(args.framerate) if args.framerate%1==0 else args.framerate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(100)

    main(args, device)
