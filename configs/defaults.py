import argparse
import numpy as np
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('SVS configs', add_help=False)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gmot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--prob_detect', default=0.2, type=float)

    # Dataset Settings
    parser.add_argument('--data_path', default='E:/dataset/MOTs', type=str)

    # Model Settings



    # Output Settings
    parser.add_argument('--out_path', default='./outputs', type=str)

    return parser