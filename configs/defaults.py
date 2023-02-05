import argparse
import numpy as np
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('SVS configs', add_help=False)

    # Dataset Settings
    parser.add_argument('--mot_path', default='E:/dataset/MOTs', type=str)
    parser.add_argument('--use_cars', action='store_true',                      help='use bounding boxes of cars too')
    parser.add_argument('--dont_cache', action='store_true',                    help='avoid storing simulated datasets in pickle files (recommended to cache)')
    
    # Configuration Setting
    parser.add_argument('--framerate', default=4, type=int)
    parser.add_argument('--svs_close', default=1, type=int)
    parser.add_argument('--svs_open', default=3, type=int)
    parser.add_argument('--svs_hot', default=5, type=int)

    # Model Settings
    parser.add_argument('--architecture', default='yolov8s', type=str,          help='yolov8s, CNN, MLP, phinet')
    parser.add_argument('--sensor', default='static', type=str,                 help='static, simple, evolution')

    # Output Settings
    parser.add_argument('--out_path', default='./outputs', type=str)
    parser.add_argument('--detect_thresh', default=0.1, type=float)

    return parser