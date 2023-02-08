from msilib.schema import Error
import cv2
import torch
from torch import nn
import numpy as np

from kornia import augmentation as K
import kornia.augmentation.functional as F

class DiffErosion(nn.Module):
    """Differentiable implementation of erosion"""
    def __init__(self, k_size):
        super().__init__()
        self.cnn = torch.nn.ConvTranspose2d(1,1,k_size, bias=False, padding=1)

        for p in self.cnn.parameters():
            p.requires_grad = False
        self.cnn.weight[0] = nn.Parameter(torch.ones((k_size,k_size),dtype=torch.float).view(1,1,k_size,k_size) / (k_size**2/2), requires_grad=False)


    def forward(self, x):
        """
            Args:
                - x: grayscale image 160x128 in range [0,1]
        """
        where_zero = 1 - x
        where_to_subtract = self.cnn(where_zero.view(-1,1,128,160))
        return torch.nn.ReLU()(1-where_to_subtract) # torch.relu(x-where_to_subtract) #


class SVSAlgorithm(nn.Module):
    """
    problems:
        1.  sigmoid params = True causes errors in backprop
    """
    def __init__(self, close=1,open=5,dhot=8, differentiable=True, sigmoid_params=False,average='exp', *a,**b):
        super().__init__()
        # non updatable params
        self.average = average
        self.differentiable = differentiable
        self.sigmoid_params = sigmoid_params
        self.HT = torch.zeros((128,160))
        self.LT = torch.zeros((128,160))
        self.erosion = DiffErosion(3)
        
        # updatable params
        params = torch.tensor([close,open,dhot], dtype=torch.float)/255
        if sigmoid_params and differentiable:
            params = torch.logit(params)
        self.backup_params = params.clone()
        self.register_parameter(name='params', param=nn.Parameter(params, requires_grad=differentiable))

        self.auto_set_bk_hook()

    def to(self, *a, **k):
        self = super().to(*a,**k)
        new_dev = None
        if 'cuda' in a or torch.device('cuda') in a:
            new_dev = torch.device('cuda')
        elif 'device' in k:
            new_dev = k['device']
        elif 'cpu' in a or torch.device('cpu') in a:
            new_dev = torch.device('cpu')
        if new_dev is not None:
            self.HT = self.HT.to(new_dev)
            self.LT = self.LT.to(new_dev)
        return self
    
    def reset_params(self, new_params=None):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.tensor(list(new_params))
        params = self.backup_params if new_params is None else new_params.float()
        if params[-1]>1: params /= 255
        if self.sigmoid_params and self.differentiable: params = torch.logit(params)
        with torch.no_grad():
            self.params.copy_(params)

    def get_params(self):
        params = self.params.clone().cpu().detach()
        if self.sigmoid_params and self.differentiable: params = torch.sigmoid(params)
        return params

    def detach_tresh(self):
        with torch.no_grad():
            self.LT = self.LT.detach()
            self.HT = self.HT.detach()
    
    def auto_set_bk_hook(self):
        self.params_counter = 1
        def quantize_gradient(grad):
            self.params_counter += 1

            alpha = 1 # default no averageing
            if self.average == 'exp':
                alpha = 0.004  # exponential averageing (c.a. 3 epochs memory)
            elif self.average == 'mean':
                alpha = 1/self.params_counter # arithmetic mean average
            
            return grad * alpha

        self.params.register_hook(quantize_gradient)
    
    def reset_counter(self, n_batches, alpha_running_avg):
        x = alpha_running_avg/(1-alpha_running_avg)
        self.params_counter = n_batches*x

    def init_thresh(self, means, stds):
        with torch.no_grad():
            self.LT.copy_(means-stds)
            self.HT.copy_(means+stds)

    def forward(self, batch_x, y=None):
        res = []
        for x in batch_x:
            if self.differentiable:
                res.append(self.forward_diff(x.squeeze(0)))
            else:
                res.append(self.forward_nondiff(x.squeeze(0)))
        if self.differentiable:
            res = torch.cat(res, dim=0)
        else:
            res = torch.stack(res).unsqueeze(1)
        return res

    def forward_diff(self, x):
        """
        Approximate SVS algorithm (output in pixel values can range between 0 and 1)

        Args:
            - x: a batch of 160x128 grayscale images (as a tensor: Shape(128,160))
        
        Returns:
            - the binary image obtained by applying the SVS algorithm
        """
        d_close, d_open, d_hot = self.params.sigmoid() if self.sigmoid_params else self.params

        tmpH_hot=torch.sigmoid((x-self.HT-d_hot)*500)
        tmpH = (x - self.HT) > 0  # pixel open
        self.HT[tmpH]  = self.HT[tmpH] + d_open  # update open
        self.HT[~tmpH] = self.HT[~tmpH] - d_close # update close


        tmpL_hot = torch.sigmoid((self.LT-x-d_hot)*500)
        tmpL = (self.LT - x) > 0  # pixel open
        self.LT[tmpL]  = self.LT[tmpL] - d_open  # update open
        self.LT[~tmpL] = self.LT[~tmpL] + d_close # update close

        HOT = tmpH_hot + tmpL_hot
        HOT = self.erosion(HOT)    # automatically batches --> shape(1,c,w,h)

        # if torch.rand(1) > 0.98:  # shows some of the inputs for yolo
        #     print(self.params)
        #     cv2.imshow('asd', HOT[0,0].clone().detach().cpu().numpy()*255)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return HOT

    def forward_nondiff(self, x):
        """
        Real SVS algorithm (output in pixel values is 0 or 1)

        Args:
            - x: a 160x128 grayscale image (as a tensor)
        
        Returns:
            - the binary image obtained by applying the SVS algorithm
        """
        device = x.device
        HOT = torch.zeros((128,160), dtype=float)

        tmpH = (x - self.HT) > 0  # pixel open
        tmpH_hot = (x - self.HT) > self.params[2]        # pixel hot
        self.HT[tmpH]  = self.HT[tmpH] + self.params[1]  # update open
        self.HT[~tmpH] = self.HT[~tmpH] - self.params[0] # update close

        tmpL = (self.LT - x) > 0  # pixel open
        tmpL_hot = (self.LT - x) > self.params[2]        # pixel hot
        self.LT[tmpL]  = self.LT[tmpL] - self.params[1]  # update open
        self.LT[~tmpL] = self.LT[~tmpL] + self.params[0] # update close

        tmp_hot = tmpH_hot | tmpL_hot
        HOT[tmp_hot] = 1
        HOT[~tmp_hot] = 0

        HOT = cv2.erode(HOT.numpy(), np.ones((3, 3), np.uint8), iterations=1)
        return torch.tensor(HOT, device=device).float()

class DiffAugmentImgs(nn.Module):
    def __init__(self,p_rot=.8, p_flip=0.5, p_tr=0.6):
        super().__init__()
        self.p_rot,self.p_flip,self.p_tr = p_rot,p_flip,p_tr
        self.vflip = lambda img: F.hflip(img)
        self.rotate = lambda img, deg: F.rotate(img, torch.tensor(deg), align_corners=True) #(img, ang_360)
        self.translate = K.RandomAffine(0, torch.tensor([.12,.12]), p=1, return_transform=True)

    def forward(self, img, labels):
        if len(img.shape)==3: img=img.unsqueeze(0)
        # H FLIP
        if np.random.rand()<self.p_flip:
            img = self.vflip(img)
            labels[:, 2] = 1-labels[:, 2]
        
        # SMALL ROTATION
        if np.random.rand()<self.p_rot:
            rot_angle = np.random.rand() * np.pi/30
            img = self.rotate(img, -rot_angle/np.pi*180)
            R = torch.tensor([[np.cos(rot_angle), -np.sin(rot_angle)],[np.sin(rot_angle), np.cos(rot_angle)]]).float()
            XY = labels[:, 2:4] - 0.5
            labels[:, 2:4] = (R @ XY.T).T   + 0.5

        # CROP SOMEWHERE
        if np.random.rand()<self.p_tr:
            img, trans = self.translate(img)
            shift = trans[0,:2,2].view(-1)
            labels[:, 2:4] = labels[:, 2:4] + shift / torch.tensor(img.shape)[[-1,-2]]

        return img, labels


class AugmentableSVSAlgorithm(nn.Module):
    """
        SVS algorithm simulator for motion detection
        can also apply data augmentation
    """
    def __init__(self, close=1,open=5,dhot=8,h_coeff=400,average='exp',*a,**b):
        super().__init__()
        self.average = average
        # non updatable params
        self.aug = DiffAugmentImgs()
        self.HT = torch.zeros((128,160))
        self.LT = torch.zeros((128,160))
        self.erosion = DiffErosion(3)
        
        # updatable params   #  open_off, close,  hot_offset
        params = torch.tensor([close,open,dhot,h_coeff*255], dtype=torch.float)/255
        self.backup_params = params.clone()
        self.register_parameter(name='params', param=nn.Parameter(params, requires_grad=True))

        # set gradient as constant
        self.auto_set_bk_hook()

    def auto_set_bk_hook(self):
        self.params_counter = 1
        def quantize_gradient(grad):
            self.params_counter += 1

            alpha = 1 # default no averageing
            if self.average == 'exp':
                alpha = 0.004  # exponential averageing (c.a. 3 epochs memory)
            elif self.average == 'mean':
                alpha = 1/self.params_counter # arithmetic mean average
            
            with torch.no_grad():
                minim = self.params[[0,0,1,0]]
                minim[0] = 1/254
                decreaseable = self.params > minim # where we can decrease the param without getting it <0
                maxim = self.params[[1,2,0,0]]
                maxim[2], maxim[3] = 160/255, 900
                increaseable = self.params < maxim
                updatable = (decreaseable*(grad>0)) + (increaseable*(grad<0))
            new_grad = ((grad>0).float() - (grad<0).float()) /255
            new_grad[3] *= grad[3]
            res =  new_grad * updatable
            res = res *alpha#/(self.params_counter)  #### this should average the params through time
            return res

        self.params.register_hook(quantize_gradient)
    
    def reset_counter(self, n_batches, alpha_running_avg):
        x = alpha_running_avg/(1-alpha_running_avg)
        self.params_counter = n_batches*x

    
    def forward(self, batch_x, y=None):
        res = []
        for x in batch_x:
            # frames need to be processed sequentially
            res.append(self.forward_one(x.squeeze(0)))
        
        if y is None:  return torch.cat(res, dim=0)
        
        ##### not the most efficient way --> TODO: use batch computation
        imgs, labels = [], []
        for i, img in enumerate(res):
            img, lab = self.aug(img,y[y[:,0]==i])
            imgs.append(img)
            labels.append(lab)
        return torch.stack(imgs),  torch.cat(labels, dim=0)
        

    def forward_one(self, x):
        """
        Approximate SVS algorithm (output in pixel values can range between 0 and 1)
        offsets from the other params
        """
        d_close, d_open, d_hot, h_coeff = self.params

        tmpH_hot=torch.sigmoid((x-self.HT-d_hot)*h_coeff)
        tmpH = (x - self.HT) > 0  # pixel open
        self.HT[tmpH]  = self.HT[tmpH] + d_open  # update open
        self.HT[~tmpH] = self.HT[~tmpH] - d_close # update close


        tmpL_hot = torch.sigmoid((self.LT-x-d_hot)*h_coeff)
        tmpL = (self.LT - x) > 0  # pixel open
        self.LT[tmpL]  = self.LT[tmpL] - d_open  # update open
        self.LT[~tmpL] = self.LT[~tmpL] + d_close # update close

        HOT = tmpH_hot + tmpL_hot
        HOT = self.erosion(HOT)    # automatically batches --> shape(1,c,w,h)

        return HOT

    def to(self, *a, **k):
        self = super().to(*a,**k)
        new_dev = None
        if 'cuda' in a or torch.device('cuda') in a:
            new_dev = torch.device('cuda')
        elif 'device' in k:
            new_dev = k['device']
        elif 'cpu' in a or torch.device('cpu') in a:
            new_dev = torch.device('cpu')
        if new_dev is not None:
            self.HT = self.HT.to(new_dev)
            self.LT = self.LT.to(new_dev)
        return self
    
    def reset_params(self, new_params=None):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.tensor(list(new_params))
        params = self.backup_params if new_params is None else new_params.float()
        if params[-1]>1: params /= 255
        with torch.no_grad():
            self.params.copy_(params)

    def get_params(self):
        return self.params.clone().cpu().detach()

    def detach_tresh(self):
        with torch.no_grad():
            self.LT = self.LT.detach()
            self.HT = self.HT.detach()

    def init_thresh(self, means, stds):
        with torch.no_grad():
            self.LT.copy_(means-stds)
            self.HT.copy_(means+stds)


if __name__ == '__main__':
    img = torch.zeros((2,1,128,160))
    img[..., 40:80, 40:80] = 1
    img[..., 100:101, 40:80] = 1
    img[..., 105:107, 40:80] = 1
    img[..., 103:107, 70:80] = 1
    img[0] += torch.rand(1,128,160)/10
    
    net = AugmentableSVSAlgorithm()
    print('init', (net.get_params()*255).tolist())

    opt = torch.optim.SGD(net.parameters(), lr=1)


    for i in range(100):
        y = net(img, torch.rand(12,6))

        loss = torch.sum((y[0] - torch.zeros_like(img))**2)

        (loss).backward(retain_graph=i%4!=0)
        opt.step()
        opt.zero_grad()
        print('-->',  (net.get_params()*255).tolist())

        if i%4==0:
            net.detach_tresh()

