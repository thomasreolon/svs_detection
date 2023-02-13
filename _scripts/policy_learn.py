"""
part. 2
we try to learn a model: heuristics --> loss fn
the simulator will use this model to update his own parameters
"""
import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..') # allows import from home folder
from tqdm import tqdm
import cv2
import torch
import numpy as np
from copy import deepcopy

from datasets.simulation_ds import SimulationFastDataset
from configs.defaults import get_args_parser
from utils import init_seeds, StatsLogger
from simulators.evolutive import EvolvedSVS
from models import build as build_model, ComputeLoss

PRETRAINED = 'C:/Users/Tom/Desktop/svs_detection/_outputs/phi_pretrain.pt'
POLICY = 'C:/Users/Tom/Desktop/svs_detection/_outputs/basepolicy.pt'

def main(args, device):
    # settings
    batch_size = 32
    number_forks = 2
    n_iter = 2000

    # yolo (differentiable)
    model, optimizer, loss_fn = load_pretrained(args, device)
    model.train()
    logger = StatsLogger(args)

    # simulator (non differentiable)
    save_path = f'{args.out_path}/policy.pt'
    simulator = EvolvedSVS(args.svs_close, args.svs_open, args.svs_hot, batch_size)
    simulator.pred_reward.train()
    sim_opt = torch.optim.AdamW(simulator.pred_reward.parameters(), lr=1e-3)

    torch.autograd.set_detect_anomaly(True)

    v = 4
    pbar = tqdm(range(n_iter))
    for e in pbar:
        curr_video, imgs, tgs, v = next_video(args, e<1990, v)

        # warmup simulator
        simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))
        simulator.count = -10
        [simulator(i) for i in imgs[:10]]
        tg = tgs.clone()
        tg[:,0] -= 10

        # divide video in batches of (32)
        x_gs, ys = get_svs_gt(imgs, tgs, batch_size)

        for b, (x_, y_) in enumerate(zip(x_gs, ys)):
            results = []
            start = get_state(model, simulator, optimizer)
            for fork in range(number_forks):
                simulator.count = b*batch_size
                if fork>0: set_state(start, model, simulator, optimizer)
                else: simulator.count = -99999 # always static action (0,0,0) for fork=0

                # simulate
                xt = simulate(simulator, x_)

                # train NN
                for ex in range(5):
                    oldseed = init_seeds(e*900+v*100+ex)
                    x,y = transform(xt.copy(), y_.clone())

                    x,y = x.to(device), y.to(device)
                    _, y_p, count = model(x)
                    loss, _ = loss_fn(y_p, y, count)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    optimizer.zero_grad()

                    tqdm.write(f'- {fork}: {loss.item()}')
                    init_seeds(oldseed)
                
                # save fork results
                state = get_state(model, simulator, optimizer)
                loss = loss.item()
                pred = simulator._pred
                results.append((state, loss, pred))

            # train reward predictor
            sim_loss = 0
            state, loss, pred = results[0]  # action 0 results
            for ex, (_, l, p) in enumerate(results):
                reward = (loss-l)/(1+abs(loss))*10 # reward for changing parameters
                tqdm.write(f'>> {reward}')
                sim_loss = sim_loss + (p-reward)**2
            sim_loss.backward()
            sim_opt.step()
            sim_opt.zero_grad()

            # new state: the one with smallest loss
            l = min(*[x[1] for x in results])
            state = [x[0] for x in results if x[1]==l][-1]
            set_state(state, model, simulator, optimizer)

            # log
            text = f'video:{curr_video}  loss:{l} params:{simulator.close},{simulator.open},{simulator.dhot}'
            tqdm.write(text)
            logger.log(text)

        # FINAL SAVE
        if np.random.rand()>.9 or e==n_iter-1:
            torch.save(simulator.pred_reward.state_dict(), save_path)
            torch.save(model.state_dict(), save_path+'2')


def simulate(simulator, imgs):
    x = [simulator(i) for i in imgs]
    return np.stack(x)

def get_svs_gt(imgs, tgs, bs):
    xs = []; ys = []; batches = (len(imgs)-10)//bs
    for i in range(batches):
        idx = i*bs
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
    return [m, o, (simulator.prev_state, simulator.close, 
                   simulator.open, simulator.dhot, simulator.count,
                   simulator.Threshold_H.copy(), simulator.Threshold_L.copy())]

def set_state(s, model, simulator, optim):
    model.load_state_dict(s[0])
    optim.load_state_dict(s[1])
    simulator.prev_state, simulator.close, simulator.open, simulator.dhot, \
        simulator.count, simulator.Threshold_H, simulator.Threshold_L = s[2]


def next_video(args, rnd, i):
    # get random video / framerate
    if np.random.rand()>.5:
        p = 1/((np.array([0,1,2,3,4])-args.framerate)**2+2)
        args.framerate = np.random.choice([0.5,1,2,4,10], p=p/p.sum()) if rnd else 2
        args.framerate = args.framerate if args.framerate%1>0 else int(args.framerate)
    else:
        i = (i+1) if np.random.rand()>.5 else int(np.random.rand()*77771)
    
    ds = SimulationFastDataset(args, 106)
    infos, imgs, tgs, _ =  ds[i % len(ds)]

    # load greyscale & other
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)

    return curr_video+f':{args.framerate}', imgs, tgs, i


def load_pretrained(args, device='cuda'):
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-2, betas=(0.92, 0.999))

    # Load Pretrained
    w = torch.load(PRETRAINED, map_location='cpu')
    model.load_state_dict(w, strict=False)
    return model, optimizer, loss_fn


def transform(svss, gt_boxes, aug=True):
    imgs = [] ; gts = [] ; c = 0
    for i, svs in enumerate(svss):
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
        if len(imgs)==32: break
    gts = torch.cat(gts, dim=0)
    imgs = (((torch.from_numpy(np.stack(imgs))/255) -.1)/.9).permute(0,3,1,2)
    return imgs, gts


if __name__=='__main__':
    init_seeds(291098)
    args = get_args_parser().parse_args()
    args.use_cars=True
    args.crop_svs=True
    args.exp_name='policy'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device)
