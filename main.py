from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils.visualize import StatsLogger
from models.mlp import SimpleNN
from models._loss import ComputeLoss

args = get_args_parser().parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNN().to(device)
loss_fn = ComputeLoss(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = None

logger = StatsLogger(args)
logger.save_cfg()
if not args.skip_train:
    logger.log(f'>> Training model with {sum(p.numel() for p in model.parameters())} parameters')
    dataset = FastDataset(args, True, True)
    tr_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)

    print(f'\nStart Training ({sum(p.numel() for p in model.parameters())} params)')
    model.train()
    mainpbar = tqdm(range(args.epochs))
    for epoch in mainpbar:
        loss_mean = [0,0,0,0,0]
        pbar = tqdm(tr_loader, leave=False)
        for _, imgs, gt_boxes, _ in pbar:

            # import numpy as np ; import cv2
            # svs = np.uint8((img[0] *.9 +.1).permute(1,2,0)*255)
            # for box in gt_boxes[gt_boxes[:,0]==0, 2:]:
            #     box = box.numpy() * (160,128,160,128)
            #     x1,x2 = int(box[0]-box[2]//2), int(box[0]+box[2]//2)
            #     y1,y2 = int(box[1]-box[3]//2), int(box[1]+box[3]//2)
            #     svs = cv2.rectangle(svs, (x1,y1), (x2,y2), (255))
            # cv2.imshow('gt', svs)
            # cv2.waitKey()

            # Forward
            imgs = imgs.to(device) ; gt_boxes = gt_boxes.to(device)
            y, count = model(imgs)

            # Loss
            loss, l_item = loss_fn(y, gt_boxes, count)
            lo = l_item[:,0].mean() ; lb = l_item[l_item[:,1]>=0,1].mean() ; lc = l_item[:,2].mean()

            # Step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            loss_mean[0] += loss.item()
            loss_mean[1] += lo.item()
            loss_mean[2] += lb.item()
            loss_mean[3] += lc.item()
            loss_mean[4] += 1
            text = f"e[{epoch: 4d}]: loss_tot={loss_mean[0]/loss_mean[4]:.3e} lobj={loss_mean[1]/loss_mean[4]:.3e} lbox={loss_mean[2]/loss_mean[4]:.3e}, lcnt={loss_mean[3]/loss_mean[4]:.3e}"
            pbar.set_description(text)
        logger.log(text+'\n')
        mainpbar.set_description('>>prev '+text[7:])

        # Update Learning Strategy
        if scheduler is not None: scheduler.step()
        if epoch==args.epochs//2:
            logger.log_time()
            logger.log('>> changing loss \n')
            loss_fn.gr = .5 # penalizes confidence of badly predicted BB
        if epoch==args.epochs*4//5:
            logger.log('>> changing optimizer \n')
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr*4e-2, momentum=0.9) # diminuish lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-epoch)
        if epoch==args.epochs*9//10:
            logger.log('>> changing dataset \n')
            dataset.drop(['all_videos_MOT']) # change dataset dropping MOT17/Synth videos
            tr_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)


    torch.save(model.state_dict(), args.out_path+'/model.pt')
else:
    model.load_state_dict(torch.load(args.out_path+'/model.pt', map_location='cpu'))
    model.to(device)

## eval
print('\nStart Eval')
logger.log_time()
model.eval()
for split in [False, True]:
    dataset = FastDataset(args, split, False)
    
    v_split = 'train' if split else 'test'
    logger.new_video(v_split)
    prev_video = 'None'
    with torch.no_grad():
        for infos, imgs, tgs, ids in tqdm(DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_fn, shuffle=False)):
            imgs = imgs.to(device) ; tgs = tgs.to(device)
            
            # Inference
            preds, y, counts = model(imgs)
            _, l_items = loss_fn(y, tgs, counts)
            preds = model.model[-1].postprocess(preds, args.detect_thresh, args.nms_iou)

            # Log Stats
            obj_heat = y[0][...,4].max(dim=1)[0].cpu() # b,a,h,w,6 to b,h,w
            a = (infos, imgs.cpu(), ids, preds, l_items, obj_heat, counts.cpu())
            for i, (info, img, id, pred, l_item, heat, count) in enumerate(zip(*a)):
                if split and i%20!=0: continue # speeds up video generation by skipping frames for training set

                # create infographics
                boxes = tgs[tgs[:,0]==i, 2:].cpu()
                pred = pred[:,:4].cpu()
                logger.visualize(info, img, boxes, id, pred, heat, count)

                # log loss for every video
                curr_video = info.split(';')[0] +':'+ info.split(';')[-1]
                logger.collect_stats(f'{v_split}:{curr_video}', count, pred, boxes)
                if curr_video != prev_video:
                    if i != 0:
                        loss_obj, loss_box, loss_cnt = torch.stack(cumloss, 1).mean(1).cpu().tolist()
                        logger.log(f'[{v_split}:{prev_video}] loss_obj={loss_obj:.3e} loss_box={loss_box:.3e} loss_cnt={loss_cnt:.3e}\n')
                    cumloss = [] ; prev_video = curr_video
                cumloss.append(l_item)

logger.log_time()
logger.log_stats()
logger.close()
