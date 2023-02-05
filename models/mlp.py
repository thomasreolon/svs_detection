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
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,4,7),
            nn.ReLU(),
            nn.Conv2d(4,4,7, groups=4),
            nn.Conv2d(4,4,1),
            nn.ReLU()
        )
    def forward(self, svs_img):
        y = self.model(svs_img)
        return F.adaptive_avg_pool2d(y, (16,32))



class SimpleNN(nn.Module):
    """FFNN(wholeimage), CNN(locality)  --> Detection"""
    def __init__(self) -> None:
        super().__init__()
        anchors = torch.tensor([[4,4]])

        self.model = nn.ModuleList([
            MLP(512, 1024, 1024, 3),
            EnhanceFeatures(),
            Detect(1, anchors, [6])
        ])
        self.model[-1].stride = torch.tensor([128/16, 160/32])

    def forward(self, svs_img):
        svs_img_unrolled = F.adaptive_avg_pool2d(svs_img, (16,32)).flatten(1) #B,CHW
        y1 = self.model[0](svs_img_unrolled).view(-1, 2,16,32)
        y2 = self.model[1](svs_img)
        logits = [torch.cat((y1,y2), dim=1)] #B,6,16,32
        return self.model[2](logits)

