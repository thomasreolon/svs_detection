import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect
from ._blocks import MLP, CrossConv
from .yolov5.models.yolo import check_anchor_order 

class AppendPosEmbedd(nn.Module):
    def __init__(self, pos_size=16):
        super().__init__()
        self.pos_size = pos_size
        self.p = nn.SELU()

    def forward(self, x):
        if self.pos_size>0:
            pos = self.getposemb_sincos(x, self.pos_size)
            x =  torch.cat((x,pos), dim=1)
        return x

    @staticmethod
    def getposemb_sincos(x,c):
        bs,_,h,w = x.shape

        y_embed = torch.arange(h, dtype=x.dtype).view(1,h,1).expand(bs,-1,w)
        x_embed = torch.arange(w, dtype=x.dtype).view(1,1,w).expand(bs,h,-1)

        dim_t = torch.arange(c//2, dtype=x.dtype)
        dim_t = 10000 ** (2 * dim_t.div(2, rounding_mode='trunc') / c)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(x.device)


class BlockMLPv2(nn.Module):
    def __init__(self, ch_in, ch_out, mlp_size=5):
        super().__init__()
        self.ms = mlp_size
        self.mlp = MLP(mlp_size**2, mlp_size**2, mlp_size**2, 1) # looks at whole image
        self.cnn = CrossConv(ch_in, ch_out-1)                    # looks locally
        self.lin = nn.Sequential(                                # FFNN for channels
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, 1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 1),
            nn.SiLU()
        )

    def forward(self, x):
        # forward
        y1 = self.cnn(x)
        y2 = F.adaptive_avg_pool2d(y1[:,0], (self.ms,self.ms)).flatten(1)
        y2 = self.mlp(y2).view(-1, 1, self.ms, self.ms)
        y2 = F.adaptive_avg_pool2d(y2, y1.shape[2:])
        y = torch.cat((y1,y2), dim=1)
        # y = self.lin(y)

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
        anchors = anchors if anchors else [[10,13, 16,30, 1,5], [44,44, 50,70, 60,110]]   #[scale1[w1,h1  w2,h2], scale2[w1,h1  w2,h2]]
        ch = int(8*ch_mult)
        emb = 16 if usepos else 0

        downsampler = getdownsampler(down, ch*4, mlp_size)
        modules = [
            BlockMLPv2(ch_in=ch_in,ch_out=ch,mlp_size=mlp_size),

            BlockMLPv2(ch_in=ch,ch_out=ch*2,mlp_size=mlp_size+1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BlockMLPv2(ch_in=emb+ch*2,ch_out=ch*4,mlp_size=mlp_size),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            downsampler,
            Detect(1, anchors, [ch*4, ch*4]),
        ]
        if usepos:
            modules.insert(3, AppendPosEmbedd(16))

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
