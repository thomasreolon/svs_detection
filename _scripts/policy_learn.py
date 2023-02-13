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

from datasets.simulation_ds import SimulationFastDataset
from configs.defaults import get_args_parser
from utils import init_seeds, StatsLogger
from simulators.evolutive import EvolvedSVS
from models import build as build_model, ComputeLoss

PRETRAINED = 'C:/Users/Tom/Desktop/svs_detection/_outputs/phi_pretrain.pt'
POLICY = 'C:/Users/Tom/Desktop/svs_detection/_outputs/basepolicy.pt'

def main(args, device):
    # yolo (differentiable)
    model, optimizer, loss_fn = load_pretrained(args, device)
    model.train()
    logger = StatsLogger(args)

    # simulator (non differentiable)
    save_path = f'{args.out_path}/policy.pt'
    simulator = EvolvedSVS(args.svs_close, args.svs_open, args.svs_hot, 16)
    sim_output = [None]
    def fw_hook(module, inp, out):
        sim_output[0] = out
    simulator.pred_reward.register_forward_hook(fw_hook)
    sim_opt = torch.optim.AdamW(simulator.pred_reward.parameters(), lr=1e-3)

    v = 4
    pbar = tqdm(range(2000))
    for e in pbar:
        curr_video, imgs, tgs, v = next_video(args, e<1990, v)
        print(curr_video)
        simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))

        # warmup simulator
        [simulator(i) for i in imgs[:4]]

        svss = [] ; prev_loss = None
        simulator.count = -16
        for i, img in enumerate(imgs[4:]):
            svs = simulator(img)
            svss.append(svs)

            if len(svss) == 16 or i==len(imgs)-5:
                # INPUT
                x = (((torch.from_numpy(np.stack(svss)).clone()/255) -.1)/.9).permute(0,3,1,2).to(device)
                y = tgs.clone()
                y[:, 0] -= 4+i
                y = y[ (y[:,0]<16) & (y[:,0]>=0) ].to(device)

                # x, y = fast_aug(x, y)

                # UPDATE NN
                _, y_p, count = model(x)
                loss, _ = loss_fn(y_p, y, count)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                # UPDATE REWARD
                loss = loss.item()
                if prev_loss is not None:
                    pred_reward = sim_output[0]
                    real_reward = (prev_loss-loss)/prev_loss
                    sim_loss = (pred_reward-real_reward)**2
                    sim_loss.backward()
                    sim_opt.step()
                    sim_opt.zero_grad()
                prev_loss = loss

                # LOG
                text = f'video:{curr_video}  loss:{loss} params:{simulator.close},{simulator.open},{simulator.dhot}'
                pbar.set_description(text)
                logger.log(text)


        # FINAL SAVE
        torch.save(simulator.pred_reward.state_dict(), save_path)


def next_video(args, rnd, i):
    # get random video / framerate
    if np.random.rand()>.5:
        p = 1/((np.array([0,1,2,3,4])-args.framerate)**2+2)
        fps = np.random.choice([0.5,1,2,4,10], p=p/p.sum()) if rnd else 2
        args.framerate = fps
    else:
        i = (i+1) % len(ds) if np.random.rand()>.5 else int(np.random.rand()*len(ds))
    
    ds = SimulationFastDataset(args, 120)
    infos, imgs, tgs, _ =  ds[i]

    # load greyscale & other
    curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
    imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
    imgs = np.uint8(imgs)

    return curr_video+f':{fps}', imgs, tgs, i



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


def fast_aug(svss, gt_boxes):
    imgs = [] ; gts = []
    c = 0
    p = 40/len(svss)
    for i, svs in enumerate(svss):
        if np.random.rand()>p: continue
        # update idx gt
        gt = gt_boxes[gt_boxes[:,0]==i]
        gt[:,0] = c
        c+=1

        # hflip aug
        if np.random.rand()>0.7:
            svs = svs[:,::-1]
            gt[:, 2] = 1-gt[:, 2]
        # shift aug
        if np.random.rand()>0.7:
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
    imgs = (((torch.from_numpy(np.array(imgs))/255) -.1)/.9).permute(0,3,1,2)
    return imgs, gts


if __name__=='__main__':
    init_seeds(100)
    args = get_args_parser().parse_args()
    args.use_cars=True
    args.crop_svs=True
    args.exp_name='policy'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device)
