import numpy as np
import cv2
import torch
import pathlib
import os

from .forensor_sim import StaticSVS

"""
Motion History Image Simulator
grey where old motion
"""


class MHISVS(StaticSVS):
    name = 'mhi'

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        self.prev_mmap = np.zeros_like(self.Threshold_H)[:,:,None]

    def __call__(self, frame):
        # process as usual
        motion_map = super().__call__(frame)

        # update motion history map
        motion_map = np.maximum(motion_map, self.prev_mmap//2)
        self.prev_mmap = motion_map

        return motion_map
