from tqdm import tqdm
import torch
import os, gc
import numpy as np
from copy import deepcopy
from time import time

import pandas as pd

from datasets.simulation_ds import SimulationFastDataset
from configs.defaults import get_args_parser
from utils import init_seeds, StatsLogger
from simulators.rlearn2 import RLearnSVS
from models import build as build_model, ComputeLoss

def main(args, device):
    # settings
    batch_size = 42 # 32 train + 10 test
    number_forks = 4
    n_iter = args.n_iter
    logger = StatsLogger(args)

    # yolo (differentiable)
    model, optimizer, loss_fn = load_pretrained(args, device)
    model.train()

    # simulator (non differentiable)
    save_path = f"{args.out_path}/{args.architecture}_stats.csv"
    simulator = RLearnSVS(args.svs_close, args.svs_open, args.svs_hot, '', batch_size, True, True)

    # stats collector
    data = {'state_action':[], 'reward':[], 'video':[]}

    v = 4 ; t0 = time()
    pbar = tqdm(range(n_iter))
    for e in pbar:
        try:
            curr_video, imgs, tgs, v = next_video(args, v, batch_size)

            # warmup simulator
            simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
            simulator.count = -10 # when count is negative params are not updated (each processed frame increases count by 1)
            [simulator(i) for i in imgs[:10]]

            # divide video in batches of (32) ; skip first 10 frames
            x_gs, ys = get_svs_gt(imgs, tgs, batch_size)

            for b, (x_, y_) in enumerate(zip(x_gs, ys)):
                results = []
                start = get_state(model, simulator, optimizer)
                for fork in range(number_forks):
                    if fork==0: simulator.count = -99999    # negative count does not change params: always static action (0,0,0) for fork=0
                    if fork>0: 
                        set_state(start, model, simulator, optimizer)
                        simulator.count = b*batch_size

                    # simulate
                    xt = simulate(simulator, x_)

                    # overfit NN
                    n_ep = 2 #+ (e==0 and b==0)*10
                    for ex in range(n_ep):
                        oldseed = init_seeds(e*900+v*100+ex)
                        x,y = transform(xt.copy(), y_.clone(), ex!=n_ep-1, batch_size-10)

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

                        tqdm.write(f'- {fork}: {loss.item()}')
                        init_seeds(oldseed)
                    
                    # save fork results
                    state = get_state(model, simulator, optimizer)
                    loss = float(loss.item())
                    stateaction = simulator._sa

                    results.append((state, loss, stateaction))
                    gc.collect() ; torch.cuda.empty_cache()

                # train reward predictor
                _, no_change_loss, _ = results[0]  # action 0 results
                for ex, (_, l, sa) in enumerate(results):
                    reward = (no_change_loss-l)/(1e-5+abs(no_change_loss)) # % gain for changing parameters
                    data['state_action'].append(sa.tolist())
                    data['reward'].append(reward)
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
        except Exception as e: raise e
        # except Exception as e: logger.log(f'FAIL:{curr_video} : {e}\n')
        pbar.update(1)
        if time()-t0 > 60*60*8: break#stop after 8h


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
    m = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    o = deepcopy(optim.state_dict())
    return [m, o, (simulator.prev_state, simulator.er_k, simulator.close,
                   simulator.open, simulator.dhot, simulator.count,
                   simulator.Threshold_H.copy(), simulator.Threshold_L.copy())]

def set_state(s, model, simulator, optim):
    model.load_state_dict(s[0])
    optim.load_state_dict(s[1])
    simulator.prev_state, simulator.er_k, simulator.close, \
        simulator.open, simulator.dhot, simulator.count, \
        simulator.Threshold_H, simulator.Threshold_L = s[2]

def next_video(args, i, bs):
    # get random video / framerate
    if np.random.rand()>.5:
        # probably a similar framerate
        # NOTE: even if framerate do not change the sequence of video selected afterwards could be different
        p = 1/((np.array([3,4,5])-args.framerate)**2+2)
        args.framerate = int(np.random.choice([1,4,15], p=p/p.sum()))
    else:
        # probably a similar video ; otherwise a random video
        i = (i+1) if np.random.rand()>.2 else int(np.random.rand()*77771)
    
    ds = SimulationFastDataset(args, 999)  # select by framerate
    infos, imgs, tgs, _ =  ds[i % len(ds)] # select by video
    
    # get random interval of frames from video sequence
    n_batches = min((len(imgs)-10) // bs, 3)
    needed = 10 + bs*n_batches
    possible_starts = len(imgs) - needed
    i = int(np.random.rand()*possible_starts)
    
    # select that interval
    infos = infos[i:i+needed]
    imgs  = imgs[i:i+needed]
    tgs = tgs.view(-1,6) # b, cls, xc, yc, w, h
    tgs[:,0] -= i
    tgs = tgs[tgs[:,0]>=0]

    # load greyscale & other
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)

    return curr_video+f':{args.framerate}', imgs, tgs, i


def load_pretrained(args, device='cuda'):
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    loss_fn.gr = 0 # so that model that predict bad bounding box are not facilitated
    if args.onlycountingloss:
        loss_fn.hyp.update({'box': 0,'cls': 0,'obj': 0})

    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-2, betas=(0.92, 0.999))

    # Load Pretrained
    path = f'{args.out_path}/{args.pretrained}'
    if not os.path.exists(path): raise Exception(f'--pretrained="{path}" should be the baseline model')
    w = torch.load(path, map_location='cpu')
    model.load_state_dict(w, strict=False)
    return model, optimizer, loss_fn

def transform(svss, gt_boxes, aug=True, batch_size=32):
    imgs = [] ; gts = [] ; c = 0
    for i, svs in enumerate(svss):
        if aug     and i>=batch_size: continue # first 32 of 42 for train
        if not aug and i<batch_size*.9: continue # from 28 to 42 for test
        
        # update idx gt
        gt = gt_boxes[gt_boxes[:,0]==i]
        gt[:,0] = c
        c+=1

        # hflip aug
        if aug and np.random.rand()>0.5:
            svs = svs[:,::-1]
            gt[:, 2] = 1-gt[:, 2]
        # shift aug
        if aug and np.random.rand()>0.5:
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
    init_seeds(0)
    parser = get_args_parser()
    parser.add_argument('--n_iter', default=1200, type=int)
    parser.add_argument('--onlycountingloss', action='store_true')
    args = parser.parse_args()
    args.crop_svs=True
    args.dataset='all'
    args.exp_name='plogs/'+args.policy.split('.')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device)
