from tqdm import tqdm
import torch
import os, gc
import numpy as np
from copy import deepcopy
from time import time

import pandas as pd
from sklearn import linear_model
from collections import defaultdict
import torch.nn as nn

from configs.defaults import get_args_parser
from datasets.simulation_ds import SimulationFastDataset
from utils import init_seeds, StatsLogger
from utils.map import xywh2xyxy, update_map, get_map
from models import build as build_model, ComputeLoss
from models._head import Detect
from simulators import RLearnSVS
from simulators.policies import FixPredictor, NNPredictor, SVMPredictor

def make_neural_net_csv(args, device, save_path, data):
    # settings
    batch_size   = 42 # 32 train + 10 test
    n_iter = args.n_iter
    i_fps = args.framerate

    # yolo
    model, optimizer, loss_fn = load_pretrained(args, device)
    model.train()

    # simulator
    simulator = RLearnSVS(args.svs_close, args.svs_open, args.svs_hot, '', batch_size, True, True)

    # stats collector
    logger = StatsLogger(args)

    v = 4 ; t0 = time()
    idx = 0 if not len(data['idx']) else max(*data['idx'])+1
    pbar = tqdm(range(idx, n_iter))
    for e in pbar:
        gc.collect() ; torch.cuda.empty_cache()
        try:
            bs, curr_video, imgs, tgs, v = next_video(args, v, batch_size, i_fps)

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            last_mm = [simulator(i) for i in imgs[:10]] [-1]

            # divide video in batches of (32) ; skip first 10 frames
            x_gs, ys = get_svs_gt(imgs, tgs, bs)

            for b, (x_, y_) in enumerate(zip(x_gs, ys)):
                results = []
                start = get_state(model, simulator, optimizer)

                stateactions = simulator.get_stateactions(last_mm)
                for fork, stateaction in enumerate(stateactions):
                    if fork>0:  set_state(start, model, simulator, optimizer)
                    init_seeds(e*99999+b)

                    # set simulator params
                    action = stateaction[:4].astype(int)
                    simulator.close = action[0]
                    simulator.open  = action[1]
                    simulator.dhot  = action[2]
                    simulator.er_k  = action[3]

                    # simulate
                    xt = simulate(simulator, x_)

                    # fit NN
                    model.train()
                    n_ep = 7
                    for ex in range(n_ep):
                        oldseed = init_seeds(e*900+v*100+ex)
                        x,y = transform(xt.copy(), y_.clone(), train=True, train_batch=int(bs*0.762))

                        # import cv2 # show gt
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
                        loss, _ = loss_fn(y_p, y, y2_p)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        optimizer.zero_grad()

                        init_seeds(oldseed)

                    # score similar to MaP
                    model.eval()
                    with torch.no_grad():
                        oldseed = init_seeds(e*900+v*100+ex+1)
                        x,y = transform(xt.copy(), y_.clone(), train=False, train_batch=int(bs*0.762))
                        x,y = x.to(device), y.to(device)
                        pred, y_p, y2_p = model(x)
                        loss, _ = loss_fn(y_p, y, y2_p)
                        map_ = compute_map(y, pred)
                    
                    # save fork results
                    state = get_state(model, simulator, optimizer)
                    loss = float(loss.item())

                    results.append((state, map_, loss, stateaction, xt[-1]))

                # train reward predictor
                _, _, loss_base,_,_ = results[0]  # action 0 results
                for ex, (_, map_, loss_, sa, _) in enumerate(results):
                    reward = (loss_base-loss_)/(1e-5+abs(loss_base)) *100 # % gain for changing parameters
                    data['state_action'].append(sa.tolist())
                    data['map'].append(map_)
                    data['reward'].append(reward)
                    data['video'].append(curr_video)
                    data['idx'].append(idx)
                idx += 1
                # new state: the one with smallest loss
                best_map = max(*[x[1]+x[2]/100 for x in results], -1e99)
                state,l2,mm = [(x[0],x[2],x[-1]) for x in results if x[1]==best_map][0]
                set_state(state, model, simulator, optimizer)
                last_mm = mm

                text = f'video:{curr_video} idx:{idx} params:{simulator.close},{simulator.open},{simulator.dhot},{simulator.er_k} nnloss:{l2} MaP:{best_map}'
            # log
            tqdm.write(text)
            logger.log(text+'\n')

            if np.random.rand()>.9:
                # chaos for more exploration & learn to recover
                a,b,c = (np.random.rand(3)**3*20).astype(int)
                simulator.close = int(a*np.random.rand())+1
                simulator.open = max(simulator.close+1,b)
                simulator.dhot = max(simulator.open+1,c)
                simulator.er_k = int(np.random.rand()*6)

            # SAVE
            pd.DataFrame.from_dict(data).to_csv(save_path)
        # except Exception as e: raise e
        except Exception as e: logger.log(f'FAIL:{curr_video} : {e}\n')
        pbar.update(1)
        if time()-t0 > 60*60*8: break#stop after 8h

def make_blobdetector_csv(args, save_path, data):
    # settings
    batch_size = 40
    n_iter = args.n_iter *2
    i_fps = args.framerate

    # yolo
    model = build_model('blob', 1)

    # simulator 
    simulator = RLearnSVS(args.svs_close, args.svs_open, args.svs_hot, '', batch_size, True, True)

    # stats collector
    logger = StatsLogger(args)

    v = 4 ; t0 = time()
    idx = 0 if not len(data['idx']) else max(*data['idx'])+1
    pbar = tqdm(range(idx, n_iter))
    for e in pbar:
        gc.collect() ; torch.cuda.empty_cache()
        try:
            bs, curr_video, imgs, tgs, v = next_video(args, v, batch_size, i_fps)

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            last_mm = [simulator(i) for i in imgs[:10]] [-1]

            # divide video in batches of (32) ; skip first 10 frames
            x_gs, ys = get_svs_gt(imgs, tgs, bs)

            for b, (x_, y_) in enumerate(zip(x_gs, ys)):
                results = []
                start = get_state(None, simulator, None)

                stateactions = simulator.get_stateactions(last_mm)
                for fork, stateaction in enumerate(stateactions):
                    if fork>0:  set_state(start, None, simulator, None)
                    init_seeds(e*99999+b)

                    # set simulator params
                    action = stateaction[:4].astype(int)
                    simulator.close = action[0]
                    simulator.open  = action[1]
                    simulator.dhot  = action[2]
                    simulator.er_k  = action[3]

                    # simulate
                    xt = simulate(simulator, x_)
                    x = (((torch.from_numpy(np.stack(xt))/255) -.1)/.9).permute(0,3,1,2)

                    # get precision
                    yp = model(x)
                    map_ = compute_map(y_, yp)

                    # save fork results
                    state = get_state(None, simulator, None)

                    results.append((state, map_, 0, stateaction, xt[-1]))

                # train reward predictor
                _, _, ncl2,_,_ = results[0]  # action 0 results
                for ex, (_, map_, l2, sa, _) in enumerate(results):
                    reward = (ncl2-l2)/(1e-5+abs(ncl2)) *100 # % gain for changing parameters
                    data['state_action'].append(sa.tolist())
                    data['map'].append(map_)
                    data['reward'].append(reward)
                    data['video'].append(curr_video)
                    data['idx'].append(idx)
                idx += 1
                # new state: the one with smallest loss
                best_map = max(*[x[1] for x in results], -1e99)
                state,l2,mm = [(x[0],x[2],x[-1]) for x in results if x[1]==best_map][0]
                set_state(state, None, simulator, None)
                last_mm = mm

            # log
            text = f'video:{curr_video} params:{simulator.close},{simulator.open},{simulator.dhot},{simulator.er_k} nnloss:{l2} MaP:{best_map}'
            tqdm.write(text)
            logger.log(text+'\n')

            if np.random.rand()>.9:
                # chaos for more exploration & learn to recover
                a,b,c = (np.random.rand(3)**3*20).astype(int)
                simulator.close = int(a*np.random.rand())+1
                simulator.open = max(simulator.close+1,b)
                simulator.dhot = max(simulator.open+1,c)
                simulator.er_k = int(np.random.rand()*6)

            # SAVE
            if np.random.rand()>.8 or e+1==n_iter:
                pd.DataFrame.from_dict(data).to_csv(save_path)
        # except Exception as e: raise e
        except Exception as e: logger.log(f'FAIL:{curr_video} : {e}\n')
        pbar.update(1)
        if time()-t0 > 60*60*8: break

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


def simulate(simulator, imgs):
    x = [simulator(i) for i in imgs]
    return np.stack(x)

def get_svs_gt(imgs, tgs, bs):
    xs = []; ys = []; batches = (len(imgs)-10)//bs
    for i in range(batches):
        idx = i*bs +10
        # NN input
        x = imgs[idx:idx+bs]

        # NN target
        y = tgs.clone()
        y[:, 0] -= idx
        y = y[ (y[:,0]>=0) & (y[:,0]<bs) ]

        xs.append(x) ; ys.append(y)
    return xs, ys

def get_state(model, simulator, optim):
    m = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if model is not None else None
    o = deepcopy(optim.state_dict()) if optim is not None else None
    return [m, o, (simulator.er_k, simulator.close,
                   simulator.open, simulator.dhot, simulator.count,
                   simulator.Threshold_H.copy(), simulator.Threshold_L.copy())]

def set_state(s, model, simulator, optim):
    if model is not None:
        model.load_state_dict(s[0])
    if optim is not None:
        optim.load_state_dict(s[1])
    simulator.er_k, simulator.close, simulator.open, simulator.dhot, simulator.count = s[2][:-2]
    simulator.Threshold_H = s[2][-2].copy()
    simulator.Threshold_L = s[2][-1].copy()

def next_video(args, v, bs, fr):
    # get random video / framerate
    if np.random.rand()>.5:
        # probably a similar framerate
        # NOTE: even if framerate do not change the of video interval selected afterwards will probably be different
        p = 1/((np.array([2,4,6])-args.framerate)**2+2)
        args.framerate = int(np.random.choice([2,4,15], p=p/p.sum())) if np.random.rand()>.6 else fr
    else:
        # probably a similar video ; otherwise a random video
        v = (v+1) if np.random.rand()>.2 else int(np.random.rand()*77771)
    
    ds = SimulationFastDataset(args, 999)  # select by framerate
    infos, imgs, tgs, _ =  ds[v % len(ds)] # select by video
    tgs = tgs.view(-1,6) # b, cls, xc, yc, w, h
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    
    # if sequence too short use only one
    n_batches = min((len(imgs)-10) // bs, 3)
    if n_batches==0:
        imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
        imgs = np.uint8(imgs)
        return len(imgs)-10, curr_video+f':{args.framerate}', imgs, tgs, v

    # get random interval of frames from video sequence
    needed = 10 + bs*n_batches
    possible_starts = len(imgs) - needed
    i = int(np.random.rand()*possible_starts)
    
    # select that interval
    infos = infos[i:i+needed]
    imgs  = imgs[i:i+needed]
    tgs[:,0] -= i
    tgs = tgs[tgs[:,0]>=0]

    # load greyscale & other
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)

    return bs, curr_video+f':{args.framerate}', imgs, tgs, v


def load_pretrained(args, device='cuda'):
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    loss_fn.gr = 0 # so that model that predict bad bounding box are not facilitated

    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-2, betas=(0.92, 0.999))

    # Load Pretrained
    path = args.pretrained if os.path.isfile(args.pretrained) else f'{args.out_path}/{args.pretrained}'
    if not os.path.exists(path): raise Exception(f'--pretrained="{path}" should be the baseline model')
    w = torch.load(path, map_location='cpu')
    model.load_state_dict(w, strict=False)
    return model, optimizer, loss_fn

def transform(svss, gt_boxes, train=True, train_batch=32):
    imgs = [] ; gts = [] ; c = 0
    for i, svs in enumerate(svss):
        if train     and i>=train_batch: continue # first 32 of 42 for train
        if not train and i<train_batch*.9: continue # from 28 to 42 for test
        
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
    save_path=f'{args.out_path}/plogs/'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get infos about run with local search
    csv_path = save_path + args.architecture + '_stats.csv'
    data={'state_action':[], 'map':[], 'reward':[], 'video':[], 'idx':[]}
    if os.path.isfile(csv_path) and not args.reset:
        data = pd.read_csv(csv_path).to_dict('list')
        del data['Unnamed: 0']

    init_seeds(len(data['idx']))
    if args.architecture=='blob':
        make_blobdetector_csv(args, csv_path, data)
    else:
        make_neural_net_csv(args, device, csv_path, data)
    
    init_seeds(1)
    # learn policy model
    data = pd.read_csv(csv_path)
    data['state_action'] = list(map(lambda x:np.array(eval(x)), data['state_action']))
    data['reward'] = data['reward']*20

    print('creating policies')
    f1 = FixPredictor(data, 0)
    torch.save(f1, save_path + args.architecture + '_f1.pt')
    f2 = FixPredictor(data, 1)
    torch.save(f2, save_path + args.architecture + '_f2.pt')
    f3 = FixPredictor(data, 2)
    torch.save(f2, save_path + args.architecture + '_f2.pt')
    n1 = NNPredictor(data)
    torch.save(n1, save_path + args.architecture + '_n1.pt')
    n2 = NNPredictor(data, 1024, (3,2,2))
    torch.save(n2, save_path + args.architecture + '_n2.pt')
    sv = SVMPredictor(data)
    torch.save(sv, save_path + args.architecture + '_sv.pt')


