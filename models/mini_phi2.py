import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect
from .yolov5.models.yolo import check_anchor_order 
from .yolov5.models.phinet import PhiNet 


def get_phinet(ch_in):
    return PhiNet(
        res         = 160, 
        in_channels = ch_in, 
        out_layers  = [6,-1], 
        alpha       = 0.15,  
        B0          = 7,  
        beta        = 1.,  
        squeeze_excite      = False,  
        conv2d_input        = True, 
        downsampling_layers = [5, 7]
    )


class MiniYoloPHI2(nn.Module):
    """2 detection heads"""
    def __init__(self, ch_in=1):
        super().__init__()
        anchors =[[ 3,7,  5,14,  8,20],[13,31, 20,50, 36,80]]
        self.model = nn.ModuleList([
            get_phinet(ch_in),
            Detect(1, anchors, [14,28]),
        ])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = torch.tensor([128,160])
            with torch.no_grad():
                self.S = torch.tensor([x.shape[-3:-1] for x in self.forward(torch.zeros(1, ch_in, *s))[1]])
            m.stride = s / self.S  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 2)
            self.stride = m.stride

    def forward(self, x):
        x = self.model[0](x)
        return self.model[-1](x)
