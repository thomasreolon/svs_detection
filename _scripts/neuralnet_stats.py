"""
part. 1
get some heuristics about neural networks: NTK, RELU, LOSS, JACOB
we then train a net for 15 epochs and get the MaP score
--> we will use these stats to understand if the heuristics are good or not
"""

import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..') # allows import from home folder
from tqdm import tqdm
import os
import json
import gc
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import StatsLogger, init_seeds
from models import build_special as build_model, ComputeLoss
from engine import train_one_epoch, test_epoch
from utils.scores_nn import get_nn_heuristics
import utils.debugging as D

def main(args, device):
    stats = []
    save_path = f'{args.out_path}/nn_stats.json'
    configs = get_configs()

    for nn in configs:
        try:
            gc.collect() ; torch.cuda.empty_cache()
            stat = eval_architecture(args, device, nn)
            stat['architecture'] = nn
            stats.append(stat)

            with open(save_path, 'w') as ff:
                json.dump(stats, ff)
        except Exception as e:
            print(nn, e)

def eval_architecture(args, device, nn):

    # Setup Model & Loss
    model = build_model(nn).to(device)
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-5, betas=(0.92, 0.999))
    # Load Dataset
    dataset = FastDataset(args, True, True)
    tr_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    logger = StatsLogger(args)

    # Train
    init_seeds(11)
    np=sum(p.numel() for p in model.parameters())
    center_print(f'Starting Training {nn} ({np}param)', ' .')
    mainpbar = tqdm(range(args.epochs))

    heuristics = [get_nn_heuristics(model, loss_fn, tr_loader, device)+[700]]
    for epoch in mainpbar:

        # One Epoch
        debug = args.debug and epoch in {0,1,4,args.epochs//4,args.epochs//2,args.epochs-1}
        summary, start = train_one_epoch(tr_loader, model, loss_fn, optimizer, device, epoch, debug)
        mainpbar.set_description('>>prev '+summary[7:])
        if epoch+1==args.epochs//2:
            loss_fn.gr = .5

        if epoch<2:
            # Heuristics
            center_print(f'Getting Heuristics', ' .')
            loss1 = float(summary.split('loss_tot=')[1].split(' ')[0])
            heuristics.append(get_nn_heuristics(model, loss_fn, tr_loader, device,epoch+1)+[2000-loss1])

    # Test
    init_seeds(12)
    dataset = FastDataset(args, False, False)
    te_loader = DataLoader(dataset, batch_size=64, collate_fn=dataset.collate_fn, shuffle=True)

    # Inference
    center_print(f'Starting Evaluation', ' .')
    v_split = 'test'
    model.eval()
    with torch.no_grad():
        for j, (infos, imgs, tgs, ids) in enumerate(tqdm(te_loader, desc='eval stats')):
            imgs = imgs.to(device) ; tgs = tgs.to(device)
            # Inference
            preds, y, counts = model(imgs)
            _, l_items = loss_fn(y, tgs, counts)
            l_items = l_items.detach()
            preds = model.model[-1].postprocess(preds, args.detect_thresh, args.nms_iou)

            # Log Stats
            obj_heat = y[0][:,0,:,:,4].cpu() # b,a,h,w,6 to b,h,w
            a = (infos, imgs.cpu(), ids, preds, l_items, obj_heat, counts.cpu())
            for i, (info, img, id, pred, l_item, heat, count) in enumerate(zip(*a)):

                # create infographics
                boxes = tgs[tgs[:,0]==i, 1:].cpu()
                pred = pred.cpu() # sorted by confidence
                pred[:, :4] /= torch.tensor([160,128,160,128])

                # log loss for every video
                curr_video = info.split(';')[0] +':'+ info.split(';')[-1]
                logger.collect_stats(f'{v_split}:{curr_video}', count, pred, boxes)

    # Print Some Infos
    stats = logger.log_stats()
    stats['heuristics'] = heuristics
    stats['n_params'] = sum(p.numel() for p in model.parameters())
    center_print(f'arch:{nn}     ap50:{stats["D_ap50"]}   rc:{stats["T_accuracy"]}       me:{stats["C_meanerror"]}', color=2)
    return stats


def center_print(text, pattern=' ', color=0):
    # c   =    yellow       bold      purple       cyan
    color = ['\033[93m', '\033[1m', '\033[95m', '\033[96m'][int(color%4)]
    cols = os.get_terminal_size().columns
    n_pat = max((cols - len(text) -2)//(2*len(pattern)), 0)
    pattern = pattern * n_pat
    pattern2 = pattern
    for a,b in ['><', '()', '[]', '/\\']: pattern2 = pattern2.replace(a,b) 
    text = f'\n{color}{pattern} {text} {"".join(reversed(pattern2))}\033[0m\n'
    print(text)

def get_configs():
    return [
        ('yolophi', 0.33, 0.5, 0.35, 7, 1,   False, True),
        ('yolo5', 0.33, 1),
        ('yolo5', 0.66, 1),
        ('yolo5', 0.66, .5),
        ('yolo8', 0.33, 1),
        ('yolo8', 0.2 , .7),
        ('yolo8', 0.33, 0.5),
        ('yolophi', 0.33, 0.25, 0.35, 7, 1,   False, True),
        ('yolophi', 0.33, 0.5, 0.35, 7, 1,   False, False),
        ('yolophi', 0.33, 0.5, 0.5,  8, 1,   False, True),
        ('yolophi', 0.33, 0.25, 0.35, 7, 1.1, False, True),
        ('yolophi', 0.33, 0.25, 0.35, 7, 1,   True, True),
        ('yolophi', 0.33, 0.25, 0.2,  5, 1.1, False, True),
        ('yolophi', 0.33, 0.5, 0.35, 7, 0.9, False, True),
        ('mlp2', False, 1, 5, 3),  
        ('mlp2', False, 1, 5, 0),
        ('mlp2', False, 4, 5, 1),  
        ('mlp2', False, 2, 5, 2),  
        ('mlp2', True, 4, 7, 2),  
        ('mlp2', True, 2, 7, 3),  
        ('mlp1', (16,24), 16),
        ('mlp1', (16,24), 32),
        ('mlp1', (16,24), 64),
        ('mlp1', (8,12), 128),
    ]




if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.use_cars=True
    args.crop_svs=True
    args.debug=False
    args.epochs=15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(100)

    main(args, device)