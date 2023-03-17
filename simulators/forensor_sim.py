import numpy as np
import cv2

class StaticSVS():
    name = 'static'
    def __init__(self, d_close=1, d_open=3, d_hot=5, svs_ker=0):
        # Algorithm parameters
        self.kernels = self.get_kernels()
        self.er_k = svs_ker
        self.open = d_open
        self.close = d_close
        self.dhot = d_hot

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        self.Threshold_H = (init_threshold+(std/2)).astype(np.int)
        self.Threshold_L = (init_threshold-(std/2)).astype(np.int)
    
    def get_kernels(self):
        a = np.ones((3, 3), np.uint8)   # full
        b = np.zeros((3, 3), np.uint8)  # cross
        b[1,:]=1 ; b[:,1]=1
        c = np.zeros((3, 3), np.uint8)  # corner
        c[:2,:2]=1 ; c[0,0] = 0
        d = np.zeros((3, 3), np.uint8) ; d[1,:2]=1  # left
        e = np.zeros((3, 3), np.uint8) ; e[:2,1]=1  # top
        f = np.zeros((3, 3), np.uint8) ; f[1,1]=1  # none
        return [a,b,c,d,e,f]



    def __call__(self, frame):
        Threshold_H, Threshold_L = self.Threshold_H, self.Threshold_L
        open, close, dhot = self.open, self.close, self.dhot
        frame = frame[:,:,0]

        # activations "pixel became a lot brighter"
        tmpH = (frame - Threshold_H) > 0  # pixel open
        tmpH_hot = (frame - Threshold_H) > dhot  # pixel hot
        Threshold_H[tmpH] = Threshold_H[tmpH] + open  # update open
        Threshold_H[~tmpH] = Threshold_H[~tmpH] - close  # update close

        # activations "pixel became a lot darker"
        tmpL = (Threshold_L - frame) > 0  # pixel open
        tmpL_hot = (Threshold_L - frame) > dhot  # pixel hot
        Threshold_L[tmpL] = Threshold_L[tmpL] - open  # update open
        Threshold_L[~tmpL] = Threshold_L[~tmpL] + close  # update close

        # triggered pixels
        tmp_hot = tmpH_hot | tmpL_hot
        heat_map = np.zeros_like(frame)
        heat_map[tmp_hot] = 255
        heat_map = cv2.erode(heat_map, self.kernels[self.er_k], iterations=1)
        heat_map = heat_map.astype(np.uint8)

        return heat_map[:,:,None]

