import os
import cv2, numpy as np
import torch

_GLOBAL_STATUS = [False, None, None, None]


def debug_setup(path):
    path = f'{path}/debug'
    os.makedirs(path, exist_ok=True)
    _GLOBAL_STATUS[1] = path


def debug_regression(keep, pbox, tbox, skip=False):
    if _GLOBAL_STATUS[0]:
        img = 255*np.ones((128,160,1))
        i=1
        if not skip and (keep).sum():
            pbox = pbox.detach().cpu().numpy()
            tbox = tbox.detach().cpu().numpy()
            for box, n in zip([pbox[0],tbox[0],  pbox[-1],tbox[-1]],'p1,a1,p2,a2'.split(',')):
                img = cv2.putText(img, f'{n}: {box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}',   (8,5+i*10), cv2.FONT_HERSHEY_SIMPLEX, .33, (0,0,0))
                i+=1
        _GLOBAL_STATUS[2].append(('box', img))

def debug_heat_prediction(pi, tobj):
    if _GLOBAL_STATUS[0]:
        h1 = pi[..., 4].detach()[0,0,:,:,None].cpu().numpy()
        h1 = cv2.resize(h1,(160,128))[:,:,None]
        h2 = tobj.detach()[0,0,:,:,None].cpu().numpy()
        h2 = cv2.resize(h2,(160,128))[:,:,None]
        _GLOBAL_STATUS[2].append(('h_pred', h1*255))
        _GLOBAL_STATUS[2].append(('h_gt', h2*255))


def debug_visualize_gt(imgs, gt_boxes):
    if _GLOBAL_STATUS[0]:
        svs = np.uint8((imgs[0] *.9 +.1).permute(1,2,0)*255)
        for box in gt_boxes[gt_boxes[:,0]==0, 2:]:
            box = box.numpy() * (160,128,160,128)
            x1,x2 = int(box[0]-box[2]//2), int(box[0]+box[2]//2)
            y1,y2 = int(box[1]-box[3]//2), int(box[1]+box[3]//2)
            svs = cv2.rectangle(svs, (x1,y1), (x2,y2), (255))
        _GLOBAL_STATUS[2].append(('gt_vis', svs))

def should_debug(true, name):
    _GLOBAL_STATUS[0] = true
    _GLOBAL_STATUS[2] = []
    _GLOBAL_STATUS[3] = name

def flush_debug():
    n = _GLOBAL_STATUS[2].__len__()
    if n > 0:
        if n%2!=0 and n!=9:
            _GLOBAL_STATUS[2].append(('gt_vis', 255*np.ones((128,160,1))))
            n+=1
        ncol = int(n**0.5)
        nrow = n // ncol
        img = np.zeros((0,160*ncol,1))
        for i in range(nrow):
            tmp = _GLOBAL_STATUS[2][i::nrow]
            tmp = np.concatenate([x[1] for x in tmp], axis=1)
            img = np.concatenate([img, tmp], axis = 0)
        img[:,160::160] = 155 # grey border
        img[128::128,:] = 155 # grey border

        what = '-'.join([x[0] for x in _GLOBAL_STATUS[2]])
        f_path = f'{_GLOBAL_STATUS[1]}/{_GLOBAL_STATUS[3]}_{what}.jpg'
        cv2.imwrite(f_path, img)
        _GLOBAL_STATUS[0] = False
        _GLOBAL_STATUS[2] = []

        

