from tqdm import tqdm
import torch
import os, gc
import numpy as np
from time import time

import pandas as pd

from configs.defaults import get_args_parser
from datasets.simulation_ds import SimulationFastDataset
from utils import init_seeds, StatsLogger
from utils.map import xywh2xyxy, update_map, get_map
from models import build as build_model, ComputeLoss
from models._head import Detect
from simulators.rlearn import RLearnSVS
from simulators.policies import FixPredictor, NNPredictor, SVMPredictor

def make_neural_net_csv(args, device, save_path, data):
    # set up: model loss simulator
    model, loss_fn = load_pretrained(args, device)
    model.train()
    simulator = RLearnSVS(train=True)
    logger = StatsLogger(args)

    # global variables: v-->num_video; t0->time ; idx->run_number
    candidates = None
    t0 = time()
    idx = 0 if not len(data['idx']) else max(*data['idx'])+1
    pbar = tqdm(range(idx, args.n_iter))
    for e in pbar:
        gc.collect() ; torch.cuda.empty_cache()
        try:
            curr_video, imgs, tgs, idx = next_video(args, idx)

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            [simulator(i) for i in imgs[:10]]
            x_ = imgs[10:]
            y_ = tgs.clone()
            y_[:, 0] -= 10
            y_ = y_[ (y_[:,0]>=0) ]

            # local search: 10 steps
            heuristics = []
            old_best = []
            fail = False
            best_ever = [-1, None]
            for step in tqdm(range(10)):
                # neighbours
                if step == 0:
                    stateactions = get_starting_actions()
                else: ## add other configs with evolution from last_good
                    stateactions = simulator.get_stateactions(last_mm)
                stateactions += crossover(candidates)
                
                # simulations
                results = []
                start = get_state(model, simulator)
                for fork, stateaction in enumerate(stateactions):
                    if fork>0:  set_state(start, model, simulator)

                    # set simulator params
                    action = stateaction[:4].astype(int)
                    simulator.close = action[0]
                    simulator.open  = action[1]
                    simulator.dhot  = action[2]
                    simulator.er_k  = action[3]

                    # simulate
                    xt, h = simulate(simulator, x_)
                    heuristics += h

                    # train NN  & get MaP
                    map_ = train_eval_map(model, loss_fn, xt, y_, e*9000+step*100)
                    tqdm.write(f'{curr_video} [{step},{fork}]--> {action.tolist()} : {map_}\n')

                    # save fork results
                    results.append((get_state(model, simulator), map_, stateaction, xt[-1]))
                
                # next step
                results = sorted(results, key=lambda x: -x[1])
                state, map_, stateaction, last_mm = results[0]
                set_state(state, model, simulator)

                # suboptimal solutions, promote diversity
                candidates = []
                for _, m, s, _ in results[1:]:
                    sc1 = (0.1+m-map_)*10                       # score
                    sc2 = np.abs(stateaction[:4]-s[:4]).sum()   # diversity
                    candidates.append((s[:4], sc1 * sc2))
                candidates = sorted(candidates, key= lambda x: -x[1])
                candidates = [stateaction[:4], candidates[0][0], candidates[1][0]]
            
                # best ever
                if best_ever[0] < map_:
                    old_best.append([best_ever[1], best_ever[0]])
                    best_ever[0] = map_
                    best_ever[1] = stateaction[:4].astype(int).tolist()
                elif fail:break
                else: fail = True

            data['idx'].append(idx)
            data['video'].append(curr_video)
            data['params'].append(best_ever)
            data['other'].append(old_best[1:])
            data['heuristics'].append([h.tolist() for h in heuristics])

            # SAVE
            pd.DataFrame.from_dict(data).to_csv(save_path)
        # except Exception as e: raise e
        except Exception as e: logger.log(f'FAIL:{curr_video} : {e}')
        pbar.update(1)
        if time()-t0 > 60*60*8: break#stop after 8h

def compute_map(gts, y_pred):
    gts = gts.cpu()
    if not isinstance(y_pred, torch.Tensor):
        y_pred = [torch.cat((torch.from_numpy(x),torch.ones(x.shape[0],1),torch.zeros(x.shape[0],1)),dim=1) for x in y_pred]
    else:
        y_pred = y_pred.cpu()
        y_pred = Detect.postprocess(y_pred, 0.4, 0.4)
    stats = []
    for i,pred in enumerate(y_pred):
        gt = gts[gts[:,0]==i,1:]
        pred = pred/torch.tensor([160.,128,160,128,1,1])
        pred[:,:4] = xywh2xyxy(pred[:,:4])
        gt[:,1:]   = xywh2xyxy(gt[:,1:])
        update_map(pred , gt, stats)
    map = get_map(stats) [-1]
    return map

def get_starting_actions():
    return [
        np.array([1,2,3,2]),
        np.array([1,3,4,5]),
        np.array([2,4,11,2]),
        np.array([1,12,13,5]),
    ]

def crossover(candidates):
    if candidates is None: return []
    res = candidates[1:] + get_starting_actions()
    for c in candidates[1:] + get_starting_actions():
        r = np.random.rand(4) **2
        tmp = r*candidates[0] + c*(1-r)
        res.append(tmp.round())
    return res

def train_eval_map(model, loss_fn, xt, y_, base_seed):
    map_ = 0 ; fail = False ; i = 0
    optimizer = get_optim(model)
    new =  train_epoch(model, loss_fn, xt, y_, base_seed, optimizer)
    while not fail or new > map_: # continue train if improvement
        map_ = new ; i += 1
        new =  train_epoch(model, loss_fn, xt, y_, base_seed+i, optimizer)
        if new < map_: fail = True
    return map_

def train_epoch(model, loss_fn, xt, y_, base_seed, optimizer):
    # fit NN
    model.train()
    for ex in range(10):
        oldseed = init_seeds(base_seed+ex)
        x,y = transform(xt.copy(), y_.clone(), train=True)
        x,y = x.to(device), y.to(device)
        pred, y_p, y2_p = model(x)
        loss, _ = loss_fn(y_p, y, y2_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        init_seeds(oldseed)

    # score similar to MaP
    model.eval()
    with torch.no_grad():
        x,y = transform(xt.copy(), y_.clone(), train=False)
        x,y = x.to(device), y.to(device)
        pred, y_p, y2_p = model(x)
        map_ = compute_map(y, pred)
    return map_


def simulate(simulator, imgs):
    h,x = [], []
    for img in imgs:
        x.append(simulator(img))
        h.append(simulator.heuristics)
    return np.stack(x), h


def get_state(model, simulator):
    m = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if model is not None else None
    return [m, (simulator.er_k, simulator.close,
                   simulator.open, simulator.dhot, simulator.count,
                   simulator.Threshold_H.copy(), simulator.Threshold_L.copy())]

def set_state(s, model, simulator):
    if model is not None:
        model.load_state_dict(s[0])
    simulator.er_k, simulator.close, simulator.open, simulator.dhot, simulator.count = s[1][:-2]
    simulator.Threshold_H = s[1][-2].copy()
    simulator.Threshold_L = s[1][-1].copy()

def next_video(args, v):
    v += 1
    ds = SimulationFastDataset(args, 999)  # select by framerate
    infos, imgs, tgs, _ =  ds[v % len(ds)] # select by video
    tgs = tgs.view(-1,6) # b, cls, xc, yc, w, h

    # randomness in video (especially for how train/test are divided)
    i = int(np.random.rand()*15)
    tgs[:, 0] -= i
    imgs = imgs[i:]

    # to numpy
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    return curr_video+f':{args.framerate}', imgs, tgs, v

def get_optim(model):
    return torch.optim.AdamW([
        {'params': model.model[1:].parameters(), 'lr': args.lr},
        {'params': model.model[0].parameters()}
    ], lr=args.lr/3, betas=(0.92, 0.999))

def load_pretrained(args, device='cuda'):
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    loss_fn.gr = 0 # so that model that predict bad bounding box are not facilitated

    # Load Pretrained
    path = args.pretrained if os.path.isfile(args.pretrained) else f'{args.out_path}/{args.pretrained}'
    if not os.path.exists(path): raise Exception(f'--pretrained="{path}" should be the baseline model')
    w = torch.load(path, map_location='cpu')
    model.load_state_dict(w, strict=False)
    return model, loss_fn

def transform(svss, gt_boxes, train=True):
    imgs = [] ; gts = [] ; c = 0
    for i, svs in enumerate(svss):
        if train     and i%15>10: continue # [0..10] train
        if not train and i%15<13: continue # [13,14] test
        
        # update idx gt
        gt = gt_boxes[gt_boxes[:,0]==i]
        gt[:,0] = c
        c+=1

        # hflip aug
        if train and np.random.rand()>0.5:
            svs = svs[:,::-1]
            gt[:, 2] = 1-gt[:, 2]
        # shift aug
        if train and np.random.rand()>0.5:
            a,b = [int(x) for x in (np.random.rand(2) * 40 -20)]
            col = np.zeros((128,abs(a),1),dtype=np.uint8)
            if a>0:
                svs = np.concatenate((col,svs), axis=1)
                svs = svs[:,:160]
            elif a<0:
                svs = np.concatenate((svs,col), axis=1)
                svs = svs[:,-160:]
            row = np.zeros((abs(b),160,1),dtype=np.uint8)
            if b>0:
                svs = np.concatenate((row,svs), axis=0)
                svs = svs[:128]
            elif b<0:
                svs = np.concatenate((svs,row), axis=0)
                svs = svs[-128:]
            gt[:, 2] += a/160
            gt[:, 3] += b/128
            gt = gt[(gt[:,2]>0)&(gt[:,3]>0)&(gt[:,2]<1)&(gt[:,3]<1)]

        imgs.append(svs)
        gts.append(gt)
    gts = torch.cat(gts, dim=0)
    imgs = (((torch.from_numpy(np.stack(imgs))/255) -.1)/.9).permute(0,3,1,2)
    return imgs, gts

if __name__=='__main__':
    # experiment settings
    parser = get_args_parser()
    args = parser.parse_args()
    args.crop_svs=True
    save_path=f'{args.out_path}/plogs2/'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get infos about run with local search
    csv_path = save_path + args.architecture + '_stats.csv'
    data={'heuristics':[], 'params':[], 'video':[], 'other':[], 'idx':[]}
    if os.path.isfile(csv_path) and not args.reset:
        data = pd.read_csv(csv_path).to_dict('list')
        del data['Unnamed: 0']

    init_seeds(len(data['idx']))
    make_neural_net_csv(args, device, csv_path, data)
    
    init_seeds(1)
    # learn policy model
    data = pd.read_csv(csv_path)

    print('creating policies')


