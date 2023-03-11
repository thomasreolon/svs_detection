import numpy as np

from .forensor_sim import StaticSVS

"""
Motion History Image Simulator + GreyScale Image
grey where old motion
"""


class MHIGreySVS(StaticSVS):
    name = 'mhicatgrey'

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

        # add greyscale infos
        mhi_cat_grey = np.concatenate((motion_map, frame), axis=2)
        return np.uint8(mhi_cat_grey)
