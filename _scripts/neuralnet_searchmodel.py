"""
part. 2
after getting some stats with neuralnets_stats, we design a score as a linear combination of the heuristics
this script uses the score to eval nets and save on a file all the scores
--> we will train the nets with the best scores
"""


import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..') # allows import from home folder
from tqdm import tqdm
import numpy as np
import json
import gc
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import init_seeds
from models import build_special as build_model, ComputeLoss
from engine import train_one_epoch
from utils.scores_nn import get_nn_heuristics, predict_map


#### NOTE: there is a memory leak... so it becomes slow after a bit
def main(args, device):
    stats = []
    save_path = f'{args.out_path}/nn_stats.json'
    configs = get_configs(20000)

    for nn in tqdm(configs):
        try:
            gc.collect() ; torch.cuda.empty_cache()
            stat = eval_architecture(args, device, nn)
            stat['architecture'] = nn
            stats.append(stat)

            with open(save_path, 'w') as ff:
                json.dump(stats, ff)
        except Exception as e:
            print(e)
            print('fail:', nn)

def eval_architecture(args, device, nn):

    # Setup Model & Loss & Dataset
    model = build_model(nn).to(device)
    n_params=sum(p.numel() for p in model.parameters())
    if n_params > 1.3e5: raise Exception(f'too many params: {n_params}. for {nn} ')
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-5, betas=(0.92, 0.999))
    dataset = FastDataset(args, True, True)
    tr_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)

    # Train
    init_seeds(291098)
    loss = 0
    for epoch in range(2):
        summary, _ = train_one_epoch(tr_loader, model, loss_fn, optimizer, device, epoch, False)
        loss_fn.gr = .5
        loss += float(summary.split('loss_tot=')[1].split(' ')[0])
    
    ntk, _, grad = get_nn_heuristics(model, loss_fn, tr_loader, device,epoch+1)
    map_ = predict_map2(ntk, grad, loss/2)

    return {'p_map':map_, 'n_params':n_params}

def get_configs(num):
    return [get_random_architecture() for _ in range(num)]

def rand(sq=1):
    return int(np.random.rand()**sq*20)/20

def get_random_architecture():
    # yolo5/8 don't have many params... lets make them less probable
    if rand()>=.85:
        anch = ANCHORS[0][int(rand()*3)]
        return ('yolo5', 0.05+rand(2), .1+rand(2)*1.4, anch)
    elif rand()>=70/85:
        anch = ANCHORS[0][int(rand()*3)]
        return ('yolo8', 0.05+rand(2), .1+rand(2)*1.4, anch)
    elif rand()>=.5:
        anch = ANCHORS[0][int(rand()*3)]
        return ('yolophi', 0.1+rand(2), 0.05+rand(2), 0.05+rand(2), int(4+rand(2)*7), .7+rand()*.6,  rand()>.5, rand()>.5, anch)
    else:
        anch = ANCHORS[1][int(rand()*3)]
        return ('mlp2', rand()>.7, 0.2+rand(2)*20, int(2+rand(2)*32), int(rand()*4), anch)

ANCHORS = [
    [
      [[10,13, 16,30, 33,23],[30,61, 17,31, 59,119],[116,90, 156,198, 60,110]], 
      [[1,4, 4,8, 16,16, 2,2],[30,61, 17,31, 59,119] ,[116,90, 156,198, 60,110]],
      [[2,6, 8,8, 5,15],[5,15, 40,40, 20,35] ,[100,100, 40,80, 80,60]],
    ], # yolo
    [
      [[30,61, 17,31, 59,119] ,[116,90, 156,198, 60,110]],
      [[10,13, 16,30, 33,23],[30,61, 17,31, 59,119]],
      [[1,4, 4,8, 16,16, 2,2], [44,44, 10,30, 50,70, 60,110]],
    ], # mlp
]

if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.use_cars=True
    args.crop_svs=True
    args.debug=False
    args.epochs=15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(5)

    main(args, device)