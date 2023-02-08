import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect
from ._blocks import MLP, CrossConv

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



class MLPDetectorv2(nn.Module):
    """FFNN(wholeimage), CNN(locality)  --> Detection"""
    POS = 12

    def __init__(self, ch_in=1):
        super().__init__()
        anchors = [[1,4,  4,8,  16,16,  2,2]]   #[scale1[w1,h1  w2,h2], scale2[w1,h1  w2,h2]]

        self.model = nn.ModuleList([
            BlockMLPv2(ch_in=ch_in,ch_out=8,mlp_size=5),

            BlockMLPv2(ch_in=8,ch_out=16,mlp_size=6),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BlockMLPv2(ch_in=16,ch_out=32,mlp_size=5),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            Detect(1, anchors, [32]),
        ])
        self.model[-1].stride = torch.tensor([4, 4])
        self._init_weights()

    def forward(self, x):
        for m in self.model[:-1]:
            x = m(x)
        return self.model[-1]([x])

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
