import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect
from ._blocks import MLP, Conv
from .yolov5.models.yolo import check_anchor_order

class BlockMLPv2(nn.Module):
    def __init__(self, ch_in, ch_out, mlp_size=5):
        super().__init__()
        self.ms = mlp_size
        self.mlp = MLP(mlp_size**2, mlp_size**2, mlp_size**2, 1) # looks at whole image
        self.cnn = nn.Sequential(      
            Conv(ch_in, ch_in, 3),
            Conv(ch_in, ch_out-1, 3)
        )
        self.lin = nn.Sequential(                                # FFNN for channels
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, 1),
            nn.SiLU(),
        )

    def forward(self, x):
        # forward
        y1 = self.cnn(x)
        y2 = F.adaptive_avg_pool2d(y1[:,0], (self.ms,self.ms)).flatten(1)
        y2 = self.mlp(y2).view(-1, 1, self.ms, self.ms)
        y2 = F.adaptive_avg_pool2d(y2, y1.shape[2:])
        y = torch.cat((y1,y2), dim=1)
        y = self.lin(y)

        # skip conn
        if x.shape[2:] != y.shape[2:]:
            x = F.adaptive_avg_pool2d(x, y.shape[2:])
        if y.shape[1] == x.shape[1]:
            y = y+x
        else:
            common = min(y.shape[1], x.shape[1])
            tmp = y[:,:common] + x[:,:common]
            y = torch.cat((tmp, y[:,common:]), dim=1)

        return y


def getdownsampler(n, ch, m):
    n = int(n) % 4
    if n==0:
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
    if n==1:
        return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    if n==2:
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
    if n==3:
        return nn.Sequential(
            BlockMLPv2(ch_in=ch,ch_out=ch, mlp_size=m),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )



class MLPDetectorv2(nn.Module):
    """FFNN(wholeimage), CNN(locality)  --> Detection"""
    POS = 12

    def __init__(self, ch_in=1, usepos=False, ch_mult=1, mlp_size=5, down=0, anchors=None):
        super().__init__()
        anchors = anchors if anchors else [[ 3,7,  5,14,  8,20],[13,31, 20,50, 36,80]]
        ch = int(4*ch_mult)

        downsampler = getdownsampler(down, ch*4, mlp_size)
        modules = [
            BlockMLPv2(ch_in=ch_in,ch_out=ch,mlp_size=mlp_size),

            Conv(ch, ch, 3),
            Conv(ch, ch*2, 3),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            Conv(ch*2, ch*2, 3),
            Conv(ch*2, ch*4, 3),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BlockMLPv2(ch_in=ch*4,ch_out=ch*4,mlp_size=mlp_size),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            downsampler,
            Detect(1, anchors, [ch*4, ch*4]),
        ]

        self.model = nn.ModuleList(modules)
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
        self._init_weights()

    def forward(self, x):
        for m in self.model[:-2]:
            x = m(x)
        y = self.model[-2](x)
        return self.model[-1]([y, x])

    @torch.no_grad()
    def _init_weights(self):
        def get_eye(d1,d2):
            w = torch.cat([torch.eye(d2) for _ in range(int(d1/d2))], dim=0)
            return w + torch.randn_like(w) * 1e-3
        for name, param in self.named_parameters():
            if '.mlp' in name:
                if 'weight' in name: # MLP.weight
                    param.copy_(get_eye(*param.shape))
                else:                # MLP.bias
                    param.copy_(torch.zeros_like(param))
