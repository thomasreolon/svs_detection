import numpy as np
import torch
import random
import numpy as np
from .visualize import StatsLogger
from .quantize_model import quantize

def init_seeds(n):
    old_seed = (torch.seed(), np.random.get_state())
    if isinstance(n,tuple):
        torch.manual_seed(n[0])
        random.seed(n[0])
        np.random.set_state(n[1])
    else:
        n = int(max(0, min(n,2**30)))
        torch.manual_seed(n)
        random.seed(n)
        np.random.seed(n)
    return old_seed



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
