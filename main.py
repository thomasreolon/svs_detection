import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils.visualize import StatsLogger
from models.mlp import SimpleNN
from models._loss import ComputeLoss
from utils.postprocess import postprocess

args = get_args_parser().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNN().to(device)
loss_fn = ComputeLoss(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dataset = FastDataset(args, True, True)
tr_loader = DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_fn, shuffle=True)

model.train()
mean_loss = [2.3,1.]
mainpbar = tqdm(range(50))
for epoch in mainpbar:
    loss_mean = [0,0,0]
    pbar = tqdm(tr_loader, leave=False)
    for info, img, gt_boxes, gt_ids in pbar:
        img = img.to(device) ; gt_boxes = gt_boxes.to(device)
        
        pred = model(img)
        loss, l_item = loss_fn(pred, gt_boxes)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log
        loss_mean[0] += l_item[0].item()
        loss_mean[1] += l_item[1].item()
        loss_mean[2] += img.shape[0]
        pbar.set_description(f"loss_obj = {l_item[0].item()}, loss_box={l_item[1].item()}")
    loss_mean[0] /= loss_mean[2]
    loss_mean[1] /= loss_mean[2]
    mainpbar.set_description(f"prev[{mean_loss}] prev[{loss_mean[:2]}]")
    mean_loss = loss_mean[:2]
    

torch.save(model.state_dict(), args.out_path+'/model.pt')


## eval
model.eval()
logger = StatsLogger(args)
dataset = FastDataset(args, True, False)
for infos, imgs, tgs, ids in tqdm(DataLoader(dataset, batch_size=128, collate_fn=dataset.collate_fn, shuffle=False)):
    imgs = imgs.to(device) ; tgs = tgs.to(device)
    
    preds, _ = model(imgs)
    preds = postprocess(preds, args.detect_thresh)

    for i, (info, img, id, pred) in enumerate(zip(infos, imgs.cpu(), ids, preds)):
        boxes = tgs[tgs[:,0]==i, 2:].cpu()
        pred = pred[:,:4].cpu()
        logger(info, img, boxes, id, pred)


logger.cap.release()


