import argparse
import numpy as np
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('SVS configs', add_help=False)

    # Dataset Settings
    parser.add_argument('--mot_path', default='E:/dataset/MOTs', type=str)
    parser.add_argument('--use_cars', action='store_true',                      help='use bounding boxes of cars too')
    parser.add_argument('--crop_svs', action='store_true',                      help='computes svs in high res, then crops it')
    parser.add_argument('--dont_cache', action='store_true',                    help='avoid storing simulated datasets in pickle files (recommended to cache)')
    
    # Model Settings
    parser.add_argument('--architecture', default='yolophi', type=str,             help='mlp, mlp2, yolo5, yolophi, yolo8')
    parser.add_argument('--simulator', default='static', type=str,              help='static, simple, policy')
    parser.add_argument('--pretrained', default='<auto>', type=str,             help='path to checkpoint, auto will search in output+expname path')
    parser.add_argument('--policy', default='policy_104.pt', type=str,             help='path to policy, used only if simulator==policy')

    # Configuration Setting
    parser.add_argument('--framerate', default=2, type=float)
    parser.add_argument('--svs_close', default=1, type=int)
    parser.add_argument('--svs_open', default=3, type=int)
    parser.add_argument('--svs_hot', default=5, type=int)

    # Training Setting
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--epochs', default=50, type=int)

    # Eval Settings
    parser.add_argument('--out_path', default='./_outputs', type=str)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--detect_thresh', default=0.5, type=float)
    parser.add_argument('--nms_iou', default=0.3, type=float)
    parser.add_argument('--debug', action='store_true',                         help='generates visuals to understand training')
    parser.add_argument('--skip_train', action='store_true',                    help='tries to load a pretrained model')
    parser.add_argument('--triggering', action='store_true',                    help='modifies dataset to improve rilevation performances  (unbalanced dataset)')

    return parser
