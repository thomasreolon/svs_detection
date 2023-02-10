import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils.debugging as D

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
        preds = model.model[-1].postprocess(preds, args.detect_thresh, args.nms_iou)

        # Log Stats
        obj_heat = y[0][:,0,:,:,4].cpu() # b,a,h,w,6 to b,h,w
        a = (infos, imgs.cpu(), ids, preds, l_items, obj_heat, counts.cpu())
        for i, (info, img, id, pred, l_item, heat, count) in enumerate(zip(*a)):
            if is_train and i%20!=0: continue # speeds up video generation by skipping frames for training set

            # create infographics
            boxes = tgs[tgs[:,0]==i, 2:].cpu()
            pred = pred[:,:4].cpu()
            logger.visualize(info, img, boxes, id, pred, heat, count)

            # log loss for every video
            curr_video = info.split(';')[0] +':'+ info.split(';')[-1]
            logger.collect_stats(f'{v_split}:{curr_video}', count, pred, boxes)
            if curr_video != prev_video:
                if j != 0:
                    loss_obj, loss_box, loss_cnt = torch.stack(cumloss, 1).mean(1).cpu().tolist()
                    logger.log(f'[{v_split}:{prev_video}] loss_obj={loss_obj:.3e} loss_box={loss_box:.3e} loss_cnt={loss_cnt:.3e}\n')
                cumloss = [] ; prev_video = curr_video
            cumloss.append(l_item)
        D.flush_debug()


