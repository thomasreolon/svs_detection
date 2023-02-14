import torch
import gc
import numpy as np
from . import xywh2xyxy
import torch.nn.functional as F

def predict_map(ntk, grd, loss):
  ntk = np.nan_to_num(np.log(ntk), nan=-10)
  return 0.00218927*ntk + -0.00143292*grd + -0.00025612*loss + 0.8031

def predict_map2(ntk, grd, loss):
  ntk = np.nan_to_num(np.log(ntk-100000), nan=-10)
  if ntk<0 or loss>300: grd*=0.8
  return -grd

def get_nn_heuristics(model, loss_fn, tr_loader, device, batch=0):
    """returns 3 scores about the network, the higher the better"""
    gc.collect(); torch.cuda.empty_cache()
    scoremodel = NNScorer(model, loss_fn, 16, device)
    tr_loader = iter(tr_loader)
    _,x,t,_ = next(tr_loader)
    for _ in range(batch):
        _,x,t,_ = next(tr_loader)
    scores =  scoremodel.score(x.to(device), t.to(device))

    #           ntk         relu        jac
    return [3e5-scores[0], scores[1], (20-scores[2])*22]

class NNScorer():
    def __init__(self, model, lossfn, samples_to_use=64, device='cpu') -> None:
        self.model = model.train()
        self.lossfn = lossfn
        self.samples_to_use = samples_to_use
        self.max_ram = (torch.cuda.get_device_properties(0).total_memory -torch.cuda.memory_allocated(0)) / 1e9
        
        self.collect_ntk = []  # https://github.com/VITA-Group/TENAS/blob/main/lib/procedures/ntk.py
        self.setup_relu(model,device) # https://github.com/BayesWatch/nas-without-training/blob/8ba0313ea1b6038e6d0c6822031a100135715e2a/score_networks.py
        gc.collect() ; torch.cuda.empty_cache()


    def score(self, x, tgs):
        "x.shape[b,1,h,w]"

        tgs_ = self.jac_gt(x, tgs)
        x.requires_grad_(True)
        self.model.K = np.zeros((x.shape[0], x.shape[0]))
        _, y, c = self.model(x) # yolo.Detect output
        _, l = self.lossfn(y, tgs, c)

        # score ReLU
        if self.model.err>0:
            print(f'errs: {self.model.err}/{self.model.tot}')
            self.model.err = 0
        _, relu_score = np.linalg.slogdet(self.model.K)
        if _==0:
            _, relu_score = np.linalg.slogdet(self.model.K+np.eye(x.shape[0]))

        # score jacobs
        jac_score = self.get_jac(x, y, tgs_)

        # score neural tangent kernel
        logit = l.sum(1)[:self.samples_to_use]
        for i,l in enumerate(logit):
            retain = not (i==len(logit)-1)
            self.update_NTK(l, retain=retain)
        ntk_score = self.get_NTK()
        return (ntk_score, relu_score, jac_score)

    def setup_relu(self, model, device):
        """maximixe"""
        model.err = 0
        model.tot = 0
        def counting_forward_hook(module, inp, out):
            try:
                if isinstance(inp, (tuple,list)):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1).detach()
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1.-x) @ (1.-x.t())
                model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
            except Exception as e:
                model.err += 1
        for _, module in model.named_modules():
            if 'ReLU' in str(type(module)):
                model.tot += 1
                module.register_forward_hook(counting_forward_hook)

    @torch.no_grad()
    def jac_gt(self, x, tgs):
        tr = []
        for i, img in enumerate(x):
            # get where target
            gt = tgs[tgs[:,0]==i]
            boxes = (gt[:,2:] * torch.tensor([img.shape[2],img.shape[1],img.shape[2],img.shape[1]],device=gt.device))
            boxes = xywh2xyxy(boxes).int().tolist()
            truth = torch.zeros_like(img).bool().squeeze(0)
            for x1,y1,x2,y2 in boxes:
                truth[y1:y2, x1:x2] = True
            tr.append(truth)
        return torch.stack(tr)
        
    def get_jac(self, x, tmpy, tgs):
        """minimize"""
        score = [] ; y=0
        for y_ in tmpy:
            y_ = F.adaptive_avg_pool2d(y_[...,4],x.shape[2:])
            y = y+y_.sum(1)
        y = y[tgs].sum()
        self.model.zero_grad()
        y.backward(retain_graph=True)

        for grad, truth in zip(x.grad, tgs):
            grad = grad.squeeze(0).detach()
            num = grad[grad>0].mean()
            num = num if not torch.isnan(num) else 2e-5
            den = grad[truth].mean().clip(min=1e-5) if truth.sum()>0 else torch.ones(1,device=y.device)
            score.append((1+num/den).log().item())
        return sum(score) / len(score)
        
    def get_jac_v2(self, x, tmpy, tgs):
        """minimize"""
        score = [] ; y=0
        for y_ in tmpy:
            y_ = F.adaptive_avg_pool2d(y_[...,4],x.shape[2:])
            y = y+y_.sum(1)
        y = y[tgs].sum()
        self.model.zero_grad()
        y.backward(retain_graph=True)

        for grad, truth in zip(x.grad, tgs):
            grad = grad.squeeze(0).detach()
            num = grad[grad>0].mean().abs()
            num = num if not torch.isnan(num) else 2e-5
            den = grad[truth].mean().abs().clip(min=1e-5) if truth.sum()>0 else torch.ones(1,device=y.device)
            score.append((1+num/den).log().item())
        return sum(score) / len(score)
        
    def update_NTK(self, logit, retain=False):
        """logit:count"""
        # compute gradient for a logit
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit), retain_graph=retain)
        params_count, max_ = 0, self.max_ram*1.25e8/(self.samples_to_use)

        # get vector of grads
        grad = []
        for name, W in self.model.named_parameters():
            if 'weight' in name and W.grad is not None:
                if params_count + W.numel() > max_:
                    print('BREAK')
                    break
                grad.append(W.grad.view(-1).detach())
                params_count += W.numel()
        self.collect_ntk.append(torch.cat(grad))

    @torch.no_grad()
    def get_NTK(self):
        """minimize"""
        # if big difference between biggest & smallest --> GOOD
        # different samples but same pattern in gradient --> easy learn
        grads = torch.stack(self.collect_ntk) # num_samples,grads
        # ntk = torch.einsum('nc,mc->nm', [grads, grads])
        ntk = grads @ grads.T
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U')
        return np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)

