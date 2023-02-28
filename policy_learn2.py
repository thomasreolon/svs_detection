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

from datasets.simulation_ds import SimulationFastDataset
from configs.defaults import get_args_parser
from utils import init_seeds, StatsLogger
from utils.map import box_iou, xywh2xyxy
from simulators.rlearn2 import RLearnSVS, FixPolicy, LinPolicy, NNPolicy
from models import build as build_model, ComputeLoss

def make_neural_net_csv(args, device, save_path):
    # settings
    batch_size   = 42 # 32 train + 10 test
    number_forks = 4
    n_iter = args.n_iter

    # yolo (differentiable)
    model, optimizer, loss_fn = load_pretrained(args, device)
    model.train()

    # simulator (non differentiable)
    simulator = RLearnSVS(args.svs_close, args.svs_open, args.svs_hot, '', batch_size, True, True)

    # stats collector
    data = {'state_action':[], 'reward':[], 'loss':[], 'improvement':[], 'video':[]}
    logger = StatsLogger(args)

    v = 4 ; t0 = time()
    pbar = tqdm(range(n_iter))
    for e in pbar:
        try:
            bs, curr_video, imgs, tgs, v = next_video(args, v, batch_size)
            simulator.updateevery = bs

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            simulator.count = -10 # when count is negative params are not updated (each processed frame increases count by 1)
            [simulator(i) for i in imgs[:10]]

            # divide video in batches of (32) ; skip first 10 frames
            x_gs, ys = get_svs_gt(imgs, tgs, bs)

            for b, (x_, y_) in enumerate(zip(x_gs, ys)):
                results = []
                start = get_state(model, simulator, optimizer)
                for fork in range(number_forks):
                    if fork==0: simulator.count = -99999    # negative count does not change params: always static action (0,0,0) for fork=0
                    if fork>0: 
                        set_state(start, model, simulator, optimizer)
                        simulator.count = 0

                    # simulate
                    xt = simulate(simulator, x_)

                    # overfit NN
                    l0 = 0
                    n_ep = 7 + (e==0 and b==0)*5
                    for ex in range(n_ep):
                        oldseed = init_seeds(e*900+v*100+ex)
                        x,y = transform(xt.copy(), y_.clone(), train=(ex!=n_ep-1), train_batch=int(bs*0.762))

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
                        _, y_p, y2_p = model(x)
                        loss, _ = loss_fn(y_p, y, y2_p)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        optimizer.zero_grad()

                        if ex==0: l0=loss.item()
                        tqdm.write(f'- {fork}: {loss.item()}')
                        init_seeds(oldseed)
                    
                    # save fork results
                    state = get_state(model, simulator, optimizer)
                    loss = float(loss.item())
                    stateaction = simulator._sa

                    results.append((state, loss, stateaction, l0-loss))
                    gc.collect() ; torch.cuda.empty_cache()

                # train reward predictor
                _, no_change_loss, _,_ = results[0]  # action 0 results
                for ex, (_, l, sa, m) in enumerate(results):
                    reward = (no_change_loss-l)/(1e-5+abs(no_change_loss)) # % gain for changing parameters
                    data['state_action'].append(sa.tolist())
                    data['reward'].append(reward)
                    data['loss'].append(l)
                    data['improvement'].append(m)
                    data['video'].append(curr_video)

                # new state: the one with smallest loss
                l = min(*[x[1] for x in results], 1e99)
                state = [x[0] for x in results if x[1]==l][-1]
                set_state(state, model, simulator, optimizer)

            # log
            text = f'video:{curr_video} params:{simulator.close},{simulator.open},{simulator.dhot},{simulator.er_k} nn_loss:{l}'
            tqdm.write(text)
            logger.log(text+'\n')

            if np.random.rand()>.8:
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

def make_blobdetector_csv(args, save_path):
    # settings
    batch_size   = 16
    number_forks = 4
    n_iter = args.n_iter *2

    # yolo (differentiable)
    model = build_model('blob', 1)

    # simulator (non differentiable)
    simulator = RLearnSVS(args.svs_close, args.svs_open, args.svs_hot, '', batch_size, True, True)

    # stats collector
    data = {'state_action':[], 'reward':[], 'loss':[], 'improvement':[], 'video':[]}
    logger = StatsLogger(args)

    v = 4 ; t0 = time()
    pbar = tqdm(range(n_iter))
    for e in pbar:
        try:
            bs, curr_video, imgs, tgs, v = next_video(args, v, batch_size)
            simulator.updateevery = bs

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            simulator.count = -10 # when count is negative params are not updated (each processed frame increases count by 1)
            [simulator(i) for i in imgs[:10]]

            # divide video in batches of (32) ; skip first 10 frames
            x_gs, ys = get_svs_gt(imgs, tgs, bs)

            for b, (x_, y_) in enumerate(zip(x_gs, ys)):
                results = []
                start = get_state(None, simulator, None)
                for fork in range(number_forks):
                    if fork==0: simulator.count = -99999    # negative count does not change params: always static action (0,0,0) for fork=0
                    if fork>0: 
                        set_state(start, None, simulator, None)
                        simulator.count = 0

                    # simulate
                    xt = simulate(simulator, x_)
                    xt = (((torch.from_numpy(np.stack(xt))/255) -.1)/.9).permute(0,3,1,2)

                    # get MaP
                    yp = model(xt)
                    loss = 1 - compute_map(y_, yp) # similar to map (blob detector confidence is always 1)


                    # save fork results
                    state = get_state(None, simulator, None)
                    stateaction = simulator._sa

                    results.append((state, loss, stateaction, 0))
                    gc.collect() ; torch.cuda.empty_cache()

                # train reward predictor
                _, no_change_loss, _,_ = results[0]  # action 0 results
                for ex, (_, l, sa, m) in enumerate(results):
                    reward = (no_change_loss-l)/(1e-5+abs(no_change_loss)) # % gain for changing parameters
                    data['state_action'].append(sa.tolist())
                    data['reward'].append(reward)
                    data['loss'].append(l)
                    data['improvement'].append(m)
                    data['video'].append(curr_video)

                # new state: the one with smallest loss
                l = min(*[x[1] for x in results], 1e99)
                state = [x[0] for x in results if x[1]==l][-1]
                set_state(state, None, simulator, None)

            # log
            text = f'video:{curr_video} params:{simulator.close},{simulator.open},{simulator.dhot},{simulator.er_k} nn_loss:{l}'
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
            if np.random.rand()>.95 or e+1==n_iter:
                pd.DataFrame.from_dict(data).to_csv(save_path)
        # except Exception as e: raise e
        except Exception as e: logger.log(f'FAIL:{curr_video} : {e}\n')
        pbar.update(1)
        if time()-t0 > 60*60*8: break

def compute_map(gts, y_pred):
    scores = []
    for i, preds in enumerate(y_pred):
        gt = xywh2xyxy(gts[gts[:,0]==i, 2:])
        preds = preds/(160,128,160,128)
        preds = torch.from_numpy(xywh2xyxy(preds))
        iou = box_iou(gt, preds) > 0.5
        detected = (iou.sum(dim=1)>0).sum() +1
        pr = detected.item() / (preds.shape[0]+1e-4)
        rc = detected.item() / (gt.shape[0]+1e-4)
        scores.append(2*pr*rc/(pr+rc))
    return sum(scores) / len(scores)


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
    return [m, o, (simulator.prev_state, simulator.er_k, simulator.close,
                   simulator.open, simulator.dhot, simulator.count,
                   simulator.Threshold_H.copy(), simulator.Threshold_L.copy())]

def set_state(s, model, simulator, optim):
    if model is not None:
        model.load_state_dict(s[0])
    if optim is not None:
        optim.load_state_dict(s[1])
    simulator.prev_state, simulator.er_k, simulator.close, \
        simulator.open, simulator.dhot, simulator.count, \
        simulator.Threshold_H, simulator.Threshold_L = s[2]

def next_video(args, v, bs):
    # get random video / framerate
    if np.random.rand()>.5:
        # probably a similar framerate
        # NOTE: even if framerate do not change the sequence of video selected afterwards could be different
        p = 1/((np.array([2,4,6])-args.framerate)**2+2)
        args.framerate = int(np.random.choice([2,4,15], p=p/p.sum()))
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


def make_linear_policy(data, save_path):
    ## Train Model
    x = np.stack([x for x in data['state_action']])
    x = x.reshape(-1,21)
    new = x[:,:4] + x[:,4:8]
    old = x[:,4:8]
    x= np.concatenate((new, old), axis=1)
    y = np.stack([x for x in data['reward']])
    reg = linear_model.LinearRegression()
    reg.fit(x,y)

    ## Save model
    coeffs = [c for c in reg.coef_]
    bias = float(reg.intercept_)

    LinPolicy(coeffs, bias)(np.zeros(21)) # test
    torch.save(['linear', coeffs, bias], save_path)

def make_nn_policy(data, save_path):
    def preproc(x):
        x = x.reshape(-1,21)
        new = x[:,:4] + x[:,4:8]
        x = np.concatenate((new, x), axis=1)
        return torch.from_numpy(x).float()

    ## Train
    model = nn.Sequential(
        nn.Linear(25, 6),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(6,1),
        )
    opt = torch.optim.Adam(model.parameters(), lr=6e-5)
    for _ in range(3001):
        batch = data.sample(256)
        x = np.stack([x for x in batch['state_action']])
        x = preproc(x)
        y = torch.from_numpy(np.stack([x for x in batch['reward']])).float()
        yp = model(x)
        loss = ((y-yp)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()

    # Save
    NNPolicy(model)(np.zeros(21)) # test
    torch.save(['nn', model], save_path)



def make_fixed_policy(data, save_path):
    # Train (find most used config)
    repeated = defaultdict(lambda: 0)
    for i in range(len(data)//4):
        batch = data[i*4:(i+1)*4]
        x_t = np.stack([x for x in batch['state_action']])
        x_t = x_t.reshape(-1,21)
        x_t =  x_t[:,:4] + x_t[:,4:8]
        y_t = np.stack([x for x in batch['reward']])
        top = np.argmax(y_t)

        # which settings are used more
        params = x_t[top]
        repeated[tuple(params.tolist())] += 1
    best_params = sorted([k for k,v in repeated.items() if v>2], key=lambda x:-x[1])[0]
    best_params = np.array(best_params)

    # Save
    FixPolicy(best_params)(np.zeros(21)) # test
    torch.save(['fix', best_params], save_path)


if __name__=='__main__':
    init_seeds(0)

    # experiment settings
    parser = get_args_parser()
    args = parser.parse_args()
    args.crop_svs=True
    save_path=f'{args.out_path}/plogs/'
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get infos about run with local search
    csv_path = save_path + args.architecture + '_stats.csv'
    if not os.path.isfile(csv_path):
        if args.architecture=='blob':
            make_blobdetector_csv(args, csv_path)
        else:
            make_neural_net_csv(args, device, csv_path)
    
    init_seeds(1)
    # learn policy model
    data = pd.read_csv(csv_path)
    data['state_action'] = list(map(lambda x:np.array(eval(x)), data['state_action']))
    data['reward'] = data['reward']*20
    make_linear_policy(
        data,
        save_path + args.architecture + '_lin.pt'
    )
    make_nn_policy(
        data,
        save_path + args.architecture + '_nn.pt'
    )
    make_fixed_policy(
        data,
        save_path + args.architecture + '_fix.pt'
    )

