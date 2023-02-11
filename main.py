from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import StatsLogger, init_seeds
from models import build as build_model, ComputeLoss
from engine import train_one_epoch, test_epoch
import utils.debugging as D


def main(args, device):
    # Setup Model & Loss
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW([
            {'params': model.model[1:].parameters(), 'lr': args.lr},
            {'params': model.model[0].parameters()}
        ], lr=args.lr/3, weight_decay=2e-5, betas=(0.92, 0.999))
    scheduler = None

    # Initialize Logger
    logger = StatsLogger(args)
    logger.save_cfg()
    model_path = f'{logger.out_path[:-10]}/model.pt'
    if args.debug: D.debug_setup(logger.out_path[:-10])

    # Load Pretrained
    pretrained = model_path if args.pretrained=='<auto>' else args.pretrained
    if os.path.exists(pretrained):
        m_w = model.state_dict()
        o_w = torch.load(pretrained, map_location='cpu')
        weights = {k:w for k,w in o_w.items() if k in m_w and m_w[k].shape==w.shape}
        text = f'>> Loaded pretrained with {abs(len(weights)-len(o_w))} mismatches' 
        logger.log(text+'\n') ; center_print(text, '   >', 1)
        model.load_state_dict(weights, strict=False)

    if not args.skip_train:
        # Train
        np=sum(p.numel() for p in model.parameters())
        center_print(f'Starting Training ({np}param)', ' .')
        logger.log(f'>> Training model with {np} parameters\n')
        
        # Load Dataset
        dataset = FastDataset(args, True, True)
        tr_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
        
        mainpbar = tqdm(range(args.epochs))
        for epoch in mainpbar:
            # One Epoch
            debug = args.debug and epoch in {0,1,4,args.epochs//4,args.epochs//2,args.epochs-1}
            summary, start = train_one_epoch(tr_loader, model, loss_fn, optimizer, device, epoch, debug)

            # Log Results
            if epoch==0:logger.log(start+'\n')
            logger.log(summary+'\n')
            mainpbar.set_description('>>prev '+summary[7:])

            # Update Learning Strategy
            if scheduler is not None: scheduler.step()
            if epoch+1==args.epochs//2:
                logger.log_time()
                logger.log('>> changing loss \n')
                loss_fn.gr = .5 # penalizes confidence of badly predicted BB (in yolo is set to 1, we use 0.1-->0.5)
            if epoch+1==args.epochs*4//5:
                logger.log('>> changing optimizer \n')
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr*.1, momentum=0.9) # diminuish lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-epoch)
            if args.triggering and epoch+1==args.epochs*9//10:
                logger.log('>> changing dataset \n')
                dataset.drop(['MOT', 'synth']) # change dataset dropping MOT17/Synth videos
                tr_loader = DataLoader(dataset, batch_size=args.batch_size//4, collate_fn=dataset.collate_fn, shuffle=True)

            torch.save(model.state_dict(), model_path)

    ## Test
    t = logger.log_time()
    center_print(f'Starting Evaluation ({t})', ' .')
    for is_train in [True, False]:
        # Load train/test dataset
        dataset = FastDataset(args, is_train, False)

        # Inference
        test_epoch(args, dataset, model, loss_fn, is_train, logger, device, args.debug)     

        # Print Some Infos
        center_print(f'Stats for Eval: {"Train" if is_train else "Test"}', '.-\'-_', 2+int(is_train))
        stats = logger.log_stats()
        center_print(str(stats), '.-\'-_', 2+int(is_train))


    # Log Results
    logger.log_time() ; logger.close()


def center_print(text, pattern=' ', color=0):
    # c   =    yellow       bold      purple       cyan
    color = ['\033[93m', '\033[1m', '\033[95m', '\033[96m'][int(color%4)]
    cols = os.get_terminal_size().columns
    n_pat = max((cols - len(text) -2)//(2*len(pattern)), 0)
    pattern = pattern * n_pat
    pattern2 = pattern
    for a,b in ['><', '()', '[]', '/\\']: pattern2 = pattern2.replace(a,b) 
    text = f'\n{color}{pattern} {text} {"".join(reversed(pattern2))}\033[0m\n'
    print(text)


if __name__=='__main__':
    args = get_args_parser().parse_args()
    args.use_cars=True
    args.crop_svs=True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(100)

    main(args, device)
