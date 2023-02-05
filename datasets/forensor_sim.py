######################################
# FORENSOR SIMULATOR
# Modified Thomas Reolon 2023
# . . . . . . . . . . . . . . . . 
# Copyright Massimo Gottardi 2018
######################################

import numpy as np
import cv2

class StaticSVS():
    def __init__(self, d_close=1, d_open=3, d_hot=5, img_shape=(128,160)):
        # Algorithm parameters
        self.erosion_kernel = np.ones((3, 3), np.uint8)
        self.open = d_open
        self.close = d_close
        self.dhot = d_hot

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        self.Threshold_H = init_threshold+(std/2).astype(np.uint8)
        self.Threshold_L = init_threshold-(std/2).astype(np.uint8)


    def __call__(self, frame):
        Threshold_H, Threshold_L = self.Threshold_H, self.Threshold_L
        open, close, dhot = self.open, self.close, self.dhot

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
        heat_map = cv2.erode(heat_map, self.erosion_kernel, iterations=1)
        heat_map = heat_map.astype(np.uint8)

        return heat_map

