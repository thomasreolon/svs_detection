import torch.nn.functional as F
import torch, torch.nn as nn

from ._blocks import Conv
from ._head import Detect
from .yolov5.models.yolo import check_anchor_order 
from .yolov5.models.phinet import PhiNet 


def get_phinet(ch_in):
    return PhiNet(
        res         = 160, 
        in_channels = ch_in, 
        out_layers  = [-1], 
        alpha       = 0.15,  
        B0          = 7,  
        beta        = 1.,  
        squeeze_excite      = False,  
        conv2d_input        = True, 
        downsampling_layers = [5, 7]
    )

class UpsamplerCNN(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=4)
        self.cnn = nn.Sequential(      
            Conv(ch_in+1, ch_out, 3),
            Conv(ch_out, ch_out, 3)
        )
    def forward(self, logits, motionmap):
        logits = self.upsample(logits)
        mmap = F.adaptive_avg_pool2d(motionmap, logits.shape[-2:])
        x = torch.cat((mmap, logits), dim=1)
        return self.cnn(x)


class MiniYoloPHI3(nn.Module):
    """one detection head + upscale"""
    def __init__(self, ch_in=1):
        super().__init__()
        anchors =[[3,8, 7,18, 15,36, 33,74]] 
        self.model = nn.ModuleList([
            get_phinet(ch_in), # --> 28 ch
            UpsamplerCNN(28, 16),
            Detect(1, anchors, [16]),
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

    def forward(self, motionmap):
        x = self.model[0](motionmap) # phinet
        if isinstance(x, list): x = x[-1]
        x = self.model[1](x, motionmap) # upscale
        return self.model[-1]([x])


