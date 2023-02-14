"""
part. 1
get some heuristics about simulator: white pixels per video, connected components, stability, ...
we then train a net for 4 epochs on separate videos to get loss values (getting meaningful MaP would mean training the net for much more time..)
--> we will use these stats to understand if the heuristics can help predict the loss
"""
import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..') # allows import from home folder
from tqdm import tqdm
import cv2
import torch
import numpy as np
import json
import os

from datasets.simulation_ds import SimulationFastDataset
from torch.utils.data import DataLoader
from configs.defaults import get_args_parser
from utils import init_seeds
from utils.scores_svs import get_scores as get_gt_scores
from simulators.rlearn import get_heuristics
from simulators.forensor_sim import StaticSVS
from models import build as build_model, ComputeLoss

PRETRAINED = 'C:/Users/Tom/Desktop/svs_detection/_outputs/phi_pretrain.pt'
DEVICE = 'cuda'

def main(args):
    save_path = f'{args.out_path}/stats.json'
    simulator = StaticSVS(args.svs_close, args.svs_open, args.svs_hot)
    model = None
    stats = [] ; done = set()

    if os.path.exists(save_path):
        with open(save_path, 'r') as fin:
            stats = json.load(fin)
            done  = {f'{f}{v}{p}' for f,v,p in zip(stats['fps'], stats['video'], stats['params'])}
            done.union({f'{f}{v}' for f,v,p in zip(stats['fps'], stats['video'], stats['params'])})
            stats = list(zip( stats['fps'], stats['video'], stats['params'], stats['scores'], stats['heuristics'], stats['loss'] ))

    num = len(stats)
    for fps in reversed([2,4,0.5,1,10]):
        # framerates
        args.framerate = fps
        dataloader = DataLoader(SimulationFastDataset(args, 80), batch_size=1, collate_fn=SimulationFastDataset.collate_fn, shuffle=False)
        for infos, imgs, tgs, _ in dataloader:
            # load one video
            n = len(imgs)
            curr_video = infos[0].split(';')[0] +':'+ infos[0].split(';')[-1]
            imgs = ((imgs*.9+.1)*255).permute(0,2,3,1) # B,H,W,C
            imgs = np.uint8(imgs)
            
            gt_boxes = tgs.clone().to(DEVICE)
            gt_boxes[:,0] -= n//7
            gt_boxes = gt_boxes[gt_boxes[:,0]>=0]

            PAR = [(1,2,3), (3,1,5), (2,4,10), (3,2,10), (2,2,3), (1,1,4)]
            all_params = [rand_param() for _ in range(7)] + [(1,3,5)] + [x for x in PAR if np.random.rand()>.5]
            if f'{fps}{curr_video}' in done: all_params = [(1,3,5)] + [x for x in PAR if np.random.rand()>.8]

            for params in all_params:
                if f'{fps}{curr_video}{params}' in done: continue
                try:
                    # process video
                    simulator.open = params[1] ; simulator.close = params[0] ; simulator.dhot = params[2]
                    simulator.init_video(imgs[::3,:,:,0].mean(axis=0), imgs[::3,:,:,0].std(axis=0))

                    svss = [simulator(i) for i in imgs]
                    svss = svss[n//7:]

                    # train NN
                    model, optimizer, loss_fn = load_pretrained(args, DEVICE, model)
                    nn_svss = (((torch.from_numpy(np.stack(svss)).clone()/255) -.1)/.9).permute(0,3,1,2)
                    nn_svss = nn_svss.to(DEVICE)
                    model.train()
                    pbar = tqdm(range(4),leave=False)
                    for _ in pbar:
                        for i in [n//5, -1, n*2//3]:
                            if i>0:
                                x, y = nn_svss[:i], gt_boxes[gt_boxes[:,0]<i]
                            else:
                                x, y = fast_aug(svss, gt_boxes.clone())
                                x = x.to(DEVICE)

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

                            _, y_p, count = model(x)
                            loss, _ = loss_fn(y_p, y, count)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.set_description(f'[{num}] train loss: {loss.item()}')

                    # get scores pt.1
                    all_scores = [] ; all_heuristic = [] ; all_losses = []
                    for i, svs in tqdm(enumerate(svss),leave=False,desc='eval'):
                        all_scores.append(get_gt_scores(svs, tgs[tgs[:,0]==i]))
                        all_heuristic.append(get_heuristics(svs))

                    # get scores pt.2
                    model.eval()
                    with torch.no_grad():
                        pred, y, count = model(nn_svss)
                        loss, l_item = loss_fn(y, gt_boxes, count)
                        lc = l_item[l_item[:,1]>=0,1].mean().item()
                        all_losses.append([loss.item()-lc*n*6//7,lc])

                    # pred = model.model[-1].postprocess(pred, args.detect_thresh, args.nms_iou)
                    # pred = pred[0].reshape(-1,6)[:,:4].cpu().int()
                    # draw = imgs[n//7].copy()
                    # for (x,y,w,h) in pred.tolist():
                    #     x1,y1,x2,y2 = x-w//2, y-h//2, x+w//2, y+h//2
                    #     draw[y1:y2, x1:x1+1] = 255
                    #     draw[y1:y2, x2:x2-1] = 255
                    #     draw[y1:y1+1, x1:x2] = 255
                    #     draw[y2:y2-1, x1:x2] = 255
                    # cv2.imshow('fr', draw)
                    # cv2.waitKey()

                    # save stats
                    all_scores = np.array(all_scores).mean(axis=0).tolist()
                    all_losses = np.array(all_losses).mean(axis=0).tolist()
                    all_heuristic = np.array(all_heuristic).mean(axis=0).tolist()
                    stats.append([fps, curr_video, params, all_scores, all_heuristic, all_losses])

                    # save on file 
                    if np.random.rand()>0.9:
                        a_fps, a_video, a_params, a_scores, a_heuristics, a_loss = zip(*stats)
                        with open(save_path, 'w') as ff:
                            json.dump({'fps':a_fps, 'video':a_video, 'params':a_params, 'scores':a_scores, 'heuristics':a_heuristics, 'loss':a_loss}, ff)
                    num += 1
                except Exception as e:
                    print('FAILED',fps,curr_video,params,e)
    # FINAL SAVE
    a_fps, a_video, a_params, a_scores, a_heuristics, a_loss = zip(*stats)
    with open(save_path, 'w') as ff:
        json.dump({'fps':a_fps, 'video':a_video, 'params':a_params, 'scores':a_scores, 'heuristics':a_heuristics, 'loss':a_loss}, ff)


def rand_param():
    if np.random.rand()>0.5:
        # smart
        params = np.random.rand(3) * (6, 7, 10) + (1,-3,0)
        params[1] = max(1,params[0]+params[1])
        params[2] = max(*params[:2]+1,params[2])
    elif np.random.rand()>0.3:
        # in scale
        params = np.random.rand(3) * (5, 10, 15) + 1
        params[2] = max(*params[:2]+1,params[2])
    elif np.random.rand()>0.5:
        # totally random
        params = 1+np.random.rand(3)**2 *10
        params[2] = max(*params[:2]+1,params[2])
    else:
        # smart 2
        params = np.random.rand(3) * (10, 1, 20) + (0,0,0)
        params[0], params[1] = params[1]**2 * params[0], (1-params[1])**2 * params[0]
        params[2] = max(*params[:2]+1,params[2])
    return tuple(int(p) for p in params)


_w = [None]
def load_pretrained(args, device='cuda', model=None):
    if model is None:
        model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-2, betas=(0.92, 0.999))

    # Load Pretrained
    if _w[0] is None:
        _w[0] = torch.load(PRETRAINED, map_location='cpu')
    model.load_state_dict(_w[0], strict=False)
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
    main(args)
