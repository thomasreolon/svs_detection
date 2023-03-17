from tqdm import tqdm
import torch
import os, gc
import numpy as np
from time import time
from copy import deepcopy
import pandas as pd
from random import shuffle

from configs.defaults import get_args_parser
from datasets.simulation_ds import SimulationFastDataset
from utils import init_seeds, StatsLogger
from utils.map import compute_map
from models import build as build_model, ComputeLoss
from models._head import Detect
from simulators.rlearn import RLearnSVS
from simulators.rlearnmhi import MHIRLearnSVS
from simulators.policies import FixPredictorV2, NNPredictorV1, FixPredictorV1

def make_neural_net_csv(args, device, save_csv, save_csv2, save_path, data, data2):
    # set up
    simulator = MHIRLearnSVS(train=True) if args.mhi else RLearnSVS(train=True)
    logger = StatsLogger(args)

    # global variables: v-->num_video; t0->time ; idx->run_number
    candidates = None
    t0 = time()
    idx = 0 if not len(data['idx']) else max(*data['idx'])+1
    pbar = tqdm(range(idx, args.n_iter))
    for e in pbar:
        try:
            curr_video, imgs, tgs, idx = next_video(args, idx)
            print('---> info:', curr_video, len(imgs))

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            [simulator(i) for i in imgs[:10]]
            x_ = imgs[10:]
            y_ = tgs.clone()
            y_[:, 0] -= 10
            y_ = y_[ (y_[:,0]>=0) ]

            # local search: 10 steps
            heuristics = []
            old_best = [] ; old_he = []
            fails_allowed = 2
            best_ever = [-1, None]
            t1 = time()
            used = {}
            bk_simulator = deepcopy(simulator)
            for step in tqdm(range(7)):
                if time()-t1 > 60*20: break
                # neighbours
                if   step == 0:
                    stateactions = get_starting_actions()
                elif step == 1: ## add other configs with evolution from last_good
                    stateactions = simulator.get_stateactions(last_mm) + candidates
                    stateactions += crossover(candidates[0], get_starting_actions())
                elif step == 2:
                    stateactions = simulator.get_stateactions(last_mm) + candidates
                else:
                    stateactions = simulator.get_stateactions(last_mm)[int(np.random.rand()*3)::3]
                    stateactions += crossover(candidates[0], get_starting_actions())
                if candidates is not None:
                    stateactions += crossover(candidates[0], candidates[1:])
                
                # simulations
                results = []
                shuffle(stateactions)
                for fork, stateaction in enumerate(stateactions):
                    configuration = stateaction[:4].astype(int)
                    if str(configuration) not in used:

                        # same start
                        init_seeds(e*10055)
                        model, loss_fn = load_pretrained(args, device)
                        simulator = deepcopy(bk_simulator)
                        gc.collect() ; torch.cuda.empty_cache()

                        # set simulator params
                        simulator.close = configuration[0]
                        simulator.open  = configuration[1]
                        simulator.dhot  = configuration[2]
                        simulator.er_k  = configuration[3]

                        # simulate
                        xt, h = simulate(simulator, x_)

                        # train NN  & get MaP
                        map_ = train_eval_map(model, loss_fn, xt, y_, e*9000+step*100)
                        used[str(configuration)] = (map_, stateaction, xt[-1], h, (xt[:8], y_[y_[:,0]<8]))

                    # save fork results
                    results.append(used[str(configuration)])
                    tqdm.write(f'{curr_video} [{step},{fork}]--> {configuration.tolist()} : {results[-1][0]}  | best: {best_ever}\n')

                    data2['map'].append(results[-1][0])
                    data2['heuristics'].append(results[-1][3])
                    data2['params'].append(configuration.tolist())
                
                # next step
                results = sorted(results, key=lambda x: -x[0])
                map_, stateaction, last_mm, heuristics, inputs = results[0]

                # suboptimal solutions, promote diversity
                candidates = []
                for m, s, _, _, _ in results[1:]:
                    sc1 = (0.1+m-map_)*10                       # score
                    sc2 = np.abs(stateaction[:4]-s[:4]).sum()   # diversity
                    candidates.append((s[:4], sc1 * sc2))
                candidates = sorted(candidates, key= lambda x: -x[1])
                candidates = [stateaction[:4], candidates[0][0], candidates[1][0]]
            
                # best ever
                if best_ever[0] < map_:
                    old_best.append([*best_ever])
                    best_ever[0] = map_
                    best_ever[1] = stateaction[:4].astype(int).tolist()
                    old_he += heuristics[::5]
                elif not fails_allowed: break
                else: fails_allowed -= 1

            data['idx'].append(idx)
            data['video'].append(curr_video)
            data['params'].append(best_ever)
            data['other'].append(old_best[1:])
            heuristics += old_he
            s = max(1, len(heuristics) // 100)
            data['heuristics'].append([h.tolist() for h in heuristics[::s]])

            _prev[0] = inputs if e%2==1 else None

            # SAVE
            pd.DataFrame.from_dict(data).to_csv(save_csv)
            pd.DataFrame.from_dict(data2).to_csv(save_csv2)
        except Exception as e: raise e
        # except Exception as e: print('FAIL',e); logger.log(f'FAIL:{curr_video} : {e}')
        pbar.update(1)
        if time()-t0 > 60*60*8: break#stop after 8h

def get_starting_actions():
    return [
        np.array([1,2,3,2]),
        np.array([1,3,4,5]),
        np.array([1,2,7,1]),
        np.array([2,4,11,2]),
        np.array([1,12,13,3]),
        np.array([1,3,20,4]),
        np.array([1,12,30,5]),
    ]

def crossover(best, candidates):
    if candidates is None: return []
    res = []
    for c in candidates:
        r = np.random.rand(4) **2
        tmp = (r*best + c*(1-r)).round()
        tmp[2] = max(tmp[2],tmp[1]+1)
        res.append(tmp)
    return res

def train_eval_map(model, loss_fn, xt, y_, base_seed):
    map_ = 0 ; new =  0.01 ; allow_fail = 2 ; i = 0
    optimizer = get_optim(model)
    while allow_fail or new > map_: # continue train if improvement
        if new <= map_: allow_fail -= 1
        map_ = new ; i += 1
        new =  train_epoch(model, loss_fn, xt, y_, base_seed+i, optimizer)
        print('-->', new)
    e1 = eval_epoch(model, xt, y_)
    new = train_epoch(model, loss_fn, xt, y_, base_seed-1, get_optim(model, 1))
    e2 = eval_epoch(model, xt, y_)
    return (e1+e2)/2

def train_epoch(model, loss_fn, xt, y_, base_seed, optimizer, scheduler=None, loops=4):
    if isinstance(optimizer, tuple):
        scheduler = optimizer[1]
        optimizer = optimizer[0]
        loops = 10
    # fit NN
    model.train()
    for ex in range(loops):
        oldseed = init_seeds(base_seed+ex)
        x,y = transform(xt.copy(), y_.clone(), train=True)
        x,y = x.to(device), y.to(device)
        for _ in range(3):
            i = max(0,int(np.random.rand()*(len(x)-64)))
            x_tmp = x[i:i+64]
            y_tmp = y.clone()
            y_tmp[:, 0] -= i
            y_tmp = y_tmp[(y_tmp[:,0]>=0) & (y_tmp[:,0]<64)]

            pred, y_p, y2_p = model(x_tmp)
            loss, _ = loss_fn(y_p, y_tmp, y2_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None: scheduler.step()
        init_seeds(oldseed)

    # score similar to MaP
    model.eval()
    with torch.no_grad():
        x,y = transform(xt.copy(), y_.clone(), train=False)
        # import cv2 # show gt
        # print(' showing',len(x),'/', len(xt),'images')
        # for j in range(len(x)):
        #     pred = (y[y[:,0]==j,2:].clone().cpu() * torch.tensor([160,128,160,128])).int()
        #     tmp = ((x[j:j+1].clone()*.9+.1)*255).permute(0,2,3,1).cpu().numpy()[0]
        #     for (xc,yc,w,h) in pred.tolist():
        #         x1,y1,x2,y2 = xc-w//2, yc-h//2, xc+w//2, yc+h//2
        #         tmp[y1:y2, x1:x1+2] = 128
        #         tmp[y1:y2, x2:x2-2] = 128
        #         tmp[y1:y1+2, x1:x2] = 128
        #         tmp[y2:y2-2, x1:x2] = 128
        #     cv2.imshow('tmp',np.uint8(tmp))
        #     cv2.waitKey()

        x,y = x.to(device), y.to(device)
        pred, y_p, y2_p = model(x)
        map_ = compute_map(y, pred)
    return map_

_prev = [None]
def eval_epoch(model, xt, y_,):
    # score similar to MaP
    tmp = init_seeds(333)

    model.eval()
    with torch.no_grad():
        # Map over test set --> base
        x,y = transform(xt.copy(), y_.clone(), train=False)
        x,y = x.to(device), y.to(device)
        pred, _, _ = model(x)
        map1 = compute_map(y, pred)

        # Map over all --> if test frames don't have detection use the overall map + how overfittable it is
        xt = (((torch.from_numpy(np.stack(xt))/255) -.1)/.9).permute(0,3,1,2)
        x,y = xt.to(device), y_.to(device)
        pred, _, _ = model(x)
        map2 = compute_map(y, pred)
        print('=>', map1, map2, end='')

        # Map prev video  + parameters work well also with previous video
        map_ = map1*0.7 + map2*0.3
        if _prev[0] is not None:
            xt = (((torch.from_numpy(np.stack(_prev[0][0]))/255) -.1)/.9).permute(0,3,1,2)
            x,y = xt.to(device), _prev[0][1].to(device)
            pred, _, _ = model(x)
            map3 = compute_map(y, pred)
            map_ = map_*0.8 + map3*0.2
            print(map3)
        else: print()

    init_seeds(tmp)
    return map_

def simulate(simulator, imgs):
    h,x = [], []
    for img in imgs:
        x.append(simulator(img))
        h.append(simulator.heuristics)
    return np.stack(x), h

_vids = []
def next_video(args, idx):
    global _vids
    idx += 1
    ds = SimulationFastDataset(args, 999)  # select by framerate
    if len(_vids)==0: _vids += [0,1,2] +sorted(list(range(len(ds))), key=lambda x:np.random.rand())
    v = _vids[idx % len(_vids)]
    infos, imgs, tgs, _ =  ds[v] # select by video
    tgs = tgs.view(-1,6) # b, cls, xc, yc, w, h

    # to numpy
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    return curr_video+f':{args.framerate}', imgs, tgs, idx

def get_optim(model, v=0):
    if v == 0:
        return torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, betas=(0.92, 0.999))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)
        return  opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)

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
        if train     and i>len(svss)*0.75: continue 
        if not train and i<min(len(svss)-10, len(svss)*0.85): continue 
        
        # update idx gt
        gt = gt_boxes[gt_boxes[:,0]==i]
        gt[:,0] = c
        c+=1

        # hflip aug
        if train and np.random.rand()>0.4:
            svs = svs[:,::-1]
            gt[:, 2] = 1-gt[:, 2]
        # shift aug
        if train:
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
    args.n_iter = min(args.n_iter, 60)
    save_path=f'{args.out_path}/plogs3_fr{args.framerate}/'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get infos about run with local search
    csv_path = save_path + args.architecture + '_stats.csv'
    data={'heuristics':[], 'params':[], 'video':[], 'other':[], 'idx':[]}
    if os.path.isfile(csv_path) and not args.reset:
        data = pd.read_csv(csv_path).to_dict('list')
        del data['Unnamed: 0']

    # get infos for policy reward
    csv_path2 = save_path + args.architecture + '_stats2.csv'
    data2  = {'heuristics':[], 'params':[], 'map':[]}
    if os.path.isfile(csv_path2) and not args.reset:
        data2 = pd.read_csv(csv_path2).to_dict('list')
        del data2['Unnamed: 0']

    init_seeds(len(data['idx']))
    make_neural_net_csv(args, device, csv_path, csv_path2, save_path, data, data2)
    
    init_seeds(1)

    # learn policy model
    data = pd.read_csv(csv_path)
    print('creating policy LS')
    for n_clusters in [1,3,8]:
        policy = FixPredictorV2(data, n_clusters)
        policy(np.zeros(4+10))
        torch.save(policy, save_path + args.architecture + f'_fix{n_clusters}.pt')

    # learn policy NN
    data = pd.read_csv(csv_path2)
    print('creating policy NN')
    for size in NNPredictorV1.sizes:
        policy = NNPredictorV1(data, size)
        policy(np.zeros(4+10))
        torch.save(policy, save_path + args.architecture + f'_nn{size}.pt')

    # learn policy: worst case best configuration
    data = pd.read_csv(csv_path)
    print('creating policy LS')
    policy = FixPredictorV1(data)
    policy(np.zeros(4+10))
    torch.save(policy, save_path + args.architecture + f'_fixwb.pt')
