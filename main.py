from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
from utils import StatsLogger, init_seeds
from models import build as build_model, ComputeLoss
from engine import train_one_epoch, test_epoch


def main(args, device):
    # Setup Model & Loss
    model = build_model(args.architecture).to(device)
    loss_fn = ComputeLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None

    # Initialize Logger
    logger = StatsLogger(args)
    logger.save_cfg()
    model_path = f'{logger.out_path[:-10]}/model.pt'

    if not (args.skip_train and os.path.exists(model_path)):
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
            summary = train_one_epoch(tr_loader, model, loss_fn, optimizer, device, epoch)

            # Log Results
            logger.log(summary+'\n')
            mainpbar.set_description('>>prev '+summary[7:])

            # Update Learning Strategy
            if scheduler is not None: scheduler.step()
            if epoch==args.epochs//2:
                logger.log_time()
                logger.log('>> changing loss \n')
                loss_fn.gr = .5 # penalizes confidence of badly predicted BB (in yolo is set to 1, we use 0.1-->0.5)
            if epoch==args.epochs*4//5:
                logger.log('>> changing optimizer \n')
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr*.1, momentum=0.9) # diminuish lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-epoch)
            if epoch==args.epochs*9//10:
                logger.log('>> changing dataset \n')
                dataset.drop(['all_videos_MOT']) # change dataset dropping MOT17/Synth videos
                tr_loader = DataLoader(dataset, batch_size=args.batch_size//4, collate_fn=dataset.collate_fn, shuffle=True)

            torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

    ## Test
    t = logger.log_time()
    center_print(f'Starting Evaluation ({t})', ' .')
    for is_train in [False, True]:
        # Load train/test dataset
        dataset = FastDataset(args, is_train, False)

        # Inference
        test_epoch(args, dataset, model, loss_fn, is_train, logger, device)        

    # Log Results
    logger.log_time() ; logger.log_stats() ; logger.close()



def center_print(text, pattern=' '):
    cols = os.get_terminal_size().columns
    n_pat = (cols - len(text) -2)//(2+len(pattern))
    pattern = pattern * n_pat
    text = f'\n\033[93m{pattern} {text} {pattern}\033[0m\n'
    print(text)


if __name__=='__main__':
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(42)

    main(args, device)
