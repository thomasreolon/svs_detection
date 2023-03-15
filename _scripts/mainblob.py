import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..')
import torch
import os

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import StatsLogger, init_seeds
from models import build as build_model
from engine import test_epoch_blob

from main import center_print

def main(args, device):
    # Setup Model & Loss
    ch_in = 1 if 'cat' not in args.simulator else 2 # mhicatgrey gives 2 channels images
    model = build_model('blob', ch_in)

    # Initialize Logger
    logger = StatsLogger(args)
    logger.save_cfg()

    ## Test
    t = logger.log_time()
    center_print(f'Starting Evaluation ({t})', ' .')
    for is_train in [True, False]:
        # Load train/test dataset
        dataset = FastDataset(args, is_train, False)

        # Inference
        test_epoch_blob(args, dataset, model, (lambda a,b,c: 0), is_train, logger, device, args.debug)     

        # Print Some Infos
        center_print(f'Stats for Eval: {"Train" if is_train else "Test"}', '.-\'-_', 2+int(is_train))
        stats = logger.log_stats()
        center_print(str(stats), '.-\'-_', 2+int(is_train))

    # Log Results
    logger.log_time() ; logger.close()



if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.architecture='blob'
    args.crop_svs=True
    if not os.path.isfile(args.policy): args.policy = f'{args.out_path}/{args.policy}'
    args.framerate = int(args.framerate) if args.framerate%1==0 else args.framerate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(100)

    main(args, device)
