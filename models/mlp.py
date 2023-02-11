import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect
from ._blocks import MLP, EnhanceFeatures

class MLPDetector(nn.Module):
    """FFNN(wholeimage), CNN(locality)  --> Detection"""
    HW = [16,24] # low dim map size
    CH = 2+14

    def __init__(self, ch_in=1, hw=None, ch=None):
        super().__init__()
        if hw: self.HW =hw
        if ch: self.CH =ch
        anchors = [[1,4,  8,8,  1,1], [30,61, 17,31, 59,119]]   #[scale1[w1,h1  w2,h2], scale2[w1,h1  w2,h2]]
        hw = self.HW[0] * self.HW[1]
        self.ch_in = ch_in

        self.model = nn.ModuleList([
            MLP(hw, 2*hw, 2*hw, 1), ### CHin_Must==1 TODO:change
            EnhanceFeatures(ch_in, (self.CH-2), self.HW),
            nn.Sequential(nn.BatchNorm2d(self.CH), nn.Conv2d(self.CH, self.CH*2, 3, 1, 1), nn.ReLU()),
            nn.Sequential(nn.BatchNorm2d(self.CH*2), nn.UpsamplingBilinear2d(scale_factor=4), nn.Conv2d(self.CH*2, self.CH*2, 3, 1, 1), nn.ReLU()),
            Detect(1, anchors, [self.CH*2, self.CH*2]),
        ])
        self.model[-1].stride = torch.tensor([128/self.HW[0], 160/self.HW[1]])

        self._init_weights()

    def forward(self, svs_img):
        h,w = self.HW
        # FFNN (this is really heavy...)
        svs_img_unrolled = F.adaptive_avg_pool2d(svs_img, (h,w)).flatten(1) #B,CHW
        y1 = self.model[0](svs_img_unrolled).view(-1, 2,h,w)

        # Multi Scale CNN (more or less...)
        y2 = self.model[1](svs_img)

        # CNN to combine features & Detection HEAD
        y_cat = torch.cat((y1,y2), dim=1) #B,6,16,32
        y_cat = self.model[2](y_cat)

        # biggerscale
        y_cat2 = self.model[3](y_cat)
        y_cat2 = torch.cat((y_cat2[:,:self.ch_in] + F.adaptive_avg_pool2d(svs_img, (h*4,w*4)), y_cat2[:,self.ch_in:]), dim=1)
        return self.model[-1]([y_cat, y_cat2])

    @torch.no_grad()
    def _init_weights(self):
        def get_eye(d1,d2):
            w = torch.cat([torch.eye(d2) for _ in range(int(d1/d2))], dim=0)
            return w + torch.randn_like(w) * 1e-3
        for name, param in self.named_parameters():
            if 'model.0' in name:
                if 'weight' in name: # MLP.weight
                    param.copy_(get_eye(*param.shape))
                else:                # MLP.bias
                    param.copy_(torch.zeros_like(param))
