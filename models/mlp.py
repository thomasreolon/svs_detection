import torch.nn.functional as F
import torch, torch.nn as nn

from ._head import Detect

class MLP(nn.Module):
    """FFNN"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class EnhanceFeatures(nn.Module):
    """CNN"""
    def __init__(self, ch_in, out_ch, hw) -> None:
        super().__init__()
        self.hw = hw
        self.cnn1 = nn.Sequential(
            nn.Conv2d(ch_in,16,7),
            nn.ReLU(),
            nn.Conv2d(16,16,7, groups=4),
            nn.Conv2d(16,out_ch//2,1),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(ch_in,16,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,3, groups=4, padding=1),
            nn.Conv2d(16,out_ch//2,1),
            nn.ReLU()
        )
    def forward(self, svs_img):
        y1 = self.cnn1(svs_img)
        y1 = F.adaptive_avg_pool2d(y1, self.hw)
        y2 = self.cnn2(F.adaptive_avg_pool2d(svs_img, self.hw))
        return torch.cat((y1,y2), dim=1)


class SimpleNN(nn.Module):
    """FFNN(wholeimage), CNN(locality)  --> Detection"""
    HW = [16,32] # low dim map size
    CH = 2+14

    def __init__(self, ch_in=1):
        super().__init__()
        anchors = [[1,4,  8,8,  1,1]]   #[scale1[w1,h1  w2,h2], scale2[w1,h1  w2,h2]]
        hw = self.HW[0] * self.HW[1]

        self.model = nn.ModuleList([
            MLP(hw, 2*hw, 2*hw, 3), ### CHin_Must==1 TODO:change
            EnhanceFeatures(ch_in, (self.CH-2), self.HW),
            nn.Sequential(nn.Conv2d(self.CH, self.CH*2, 3, 1, 1), nn.ReLU()),
            Detect(1, anchors, [self.CH*2]),
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
        return self.model[3]([y_cat])

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
