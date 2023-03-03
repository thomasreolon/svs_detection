import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils.debugging as D
from policy_learn2 import compute_map

def train_one_epoch(tr_loader, model, loss_fn, optimizer, device, epoch, debug):
    model.train()
    loss_mean = np.zeros((5))
    pbar = tqdm(tr_loader, leave=False)
    for i, (_, imgs, gt_boxes, _) in enumerate(pbar):
        D.should_debug(i in {0,4} and debug, f'{epoch}-{i}')
        D.debug_visualize_gt(imgs, gt_boxes)

        # Forward
        imgs = imgs.to(device) ; gt_boxes = gt_boxes.to(device)
        _, y, count = model(imgs)

        # Loss
        loss, l_item = loss_fn(y, gt_boxes, count)
        l_item = l_item.detach()
        lo = l_item[:,0].mean() ; lb = l_item[l_item[:,1]>=0,1].mean() ; lc = l_item[l_item[:,1]>=0,1].mean()
        check_not_nan(lo, lb, lc)

        # Step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        loss_mean += (loss.item(), lo.item(), lb.item(), lc.item(), 1)
        text  = f"e[{epoch: 5d}]: loss_tot={loss_mean[0]/loss_mean[4]:.3e} lobj={loss_mean[1]/loss_mean[4]:.3e} lbox={loss_mean[2]/loss_mean[4]:.3e}, lcnt={loss_mean[3]/loss_mean[4]:.3e}"
        if i==0: start = 'e[start]'+text[8:]
        pbar.set_description(text)
        D.flush_debug()
    return text, start

def train_one_epochmap(tr_loader, model, loss_fn, optimizer, device, epoch, debug, map_=0):
    model.train()
    loss_mean = np.zeros((5))
    pbar = tqdm(tr_loader, leave=False)
    maps = []
    for i, (_, imgs, gt_boxes, _) in enumerate(pbar):
        D.should_debug(i in {0,4} and debug, f'{epoch}-{i}')
        D.debug_visualize_gt(imgs, gt_boxes)

        # Forward
        imgs = imgs.to(device) ; gt_boxes = gt_boxes.to(device)
        _, y, count = model(imgs)


        # Loss
        loss, l_item = loss_fn(y, gt_boxes, count)
        if map_>0:
            with torch.no_grad():
                model.eval()
                yp, _, _ = model(imgs)
                model.train()
            m = compute_map(gt_boxes, yp)
            maps.append(m)
            loss = loss * max(1e-3, 2*(map_ - m))
        l_item = l_item.detach()
        lo = l_item[:,0].mean() ; lb = l_item[l_item[:,1]>=0,1].mean() ; lc = l_item[l_item[:,1]>=0,1].mean()
        check_not_nan(lo, lb, lc)

        # Step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        avgmap = sum(maps) / len(maps) if len(maps) else -1
        loss_mean += (loss.item(), lo.item(), lb.item(), lc.item(), 1)
        text  = f"e[{epoch: 5d}]: loss_tot={loss_mean[0]/loss_mean[4]:.3e} map={avgmap} lobj={loss_mean[1]/loss_mean[4]:.3e} lbox={loss_mean[2]/loss_mean[4]:.3e}, lcnt={loss_mean[3]/loss_mean[4]:.3e}"
        if i==0: start = 'e[start]'+text[8:]
        pbar.set_description(text)
        D.flush_debug()
    return text, start, avgmap


def check_not_nan(*a):
    tmp = [torch.isnan(x).sum()>0 for x in a]
    if any(tmp):
        print(' --'.join([f'{n} isnan:{v}' for n,v in zip(['obj', 'box', 'cnt'],tmp)]))    
        exit(1)


@torch.no_grad()
def test_epoch(args, dataset, model, loss_fn, is_train, logger, device, debug):
    v_split = 'train' if is_train else 'test'
    logger.new_video(v_split)
    model.eval()

    prev_video = 'None'
    for j, (infos, imgs, tgs, ids) in enumerate(tqdm(DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_fn, shuffle=False))):
        D.should_debug(debug, v_split+str(j))
        D.debug_visualize_gt(imgs, tgs)
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
            if is_train and i%10!=0: continue # speeds up video generation by skipping frames for training set

            # create infographics
            boxes = tgs[tgs[:,0]==i, 1:].cpu()
            pred = pred.cpu()
            logger.visualize(info, img, boxes[:,1:], id, pred[:,:4], heat, count)

            # log loss for every video
            curr_video = info.split(';')[0] +':'+ info.split(';')[-1]
            pred[:, :4] /= torch.tensor([160,128,160,128])
            logger.collect_stats(f'{v_split}:{curr_video}', count, pred, boxes)
            if curr_video != prev_video:
                if j != 0:
                    loss_obj, loss_box, loss_cnt = torch.stack(cumloss, 1).mean(1).cpu().tolist()
                    logger.log(f'[{v_split}:{prev_video}] loss_obj={loss_obj:.3e} loss_box={loss_box:.3e} loss_cnt={loss_cnt:.3e}\n')
                cumloss = [] ; prev_video = curr_video
            cumloss.append(l_item)
        D.flush_debug()


@torch.no_grad()
def test_epoch_blob(args, dataset, model, loss_fn, is_train, logger, device, debug):
    v_split = 'train' if is_train else 'test'
    logger.new_video(v_split)

    for j, (infos, imgs, tgs, ids) in enumerate(tqdm(DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_fn, shuffle=False))):
        # Inference
        y = model(imgs)
        counts = torch.tensor([[int(len(l)>0), len(l)] for l in y])

        # Log Stats
        a = (infos, imgs, ids, y, counts)
        for i, (info, img, id, pred, count) in enumerate(zip(*a)):
            if is_train and i%10!=0: continue # speeds up video generation by skipping frames for training set

            # create infographics
            boxes = tgs[tgs[:,0]==i, 1:]
            pred = torch.from_numpy(pred).float()
            pred = torch.cat((pred, torch.ones(pred.shape[0],1),torch.zeros(pred.shape[0],1)), dim=1) # fake class & confidence
            logger.visualize(info, img, boxes[:,1:], id, pred[:,:4], None, count)
            # log loss for every video
            curr_video = info.split(';')[0] +':'+ info.split(';')[-1]
            pred[:, :4] /= torch.tensor([160,128,160,128])
            logger.collect_stats(f'{v_split}:{curr_video}', count, pred, boxes)

