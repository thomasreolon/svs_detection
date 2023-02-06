from .visualize import StatsLogger


def init_seeds(n):
    import torch
    torch.manual_seed(n)
    import random
    random.seed(n)
    import numpy as np
    np.random.seed(n)

