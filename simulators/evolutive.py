import numpy as np
import cv2
from .forensor_sim import StaticSVS

"""
Dynamic Simulator
perameters change with a previously learned policy
"""


class EvolvedSVS(StaticSVS):
    name = 'evolved'
    def __init__(self, d_close=1, d_open=3, d_hot=5):
        # Algorithm parameters
        self.erosion_kernel = np.ones((3, 3), np.uint8)
        self.open = d_open
        self.close = d_close
        self.dhot = d_hot
        self.ask=False

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        self.prev_state = None
        self.memory = [0,0]


    def __call__(self, frame):
        # process as usual
        heat_map = super().__call__(frame)

        # update params
        self.update_params(heat_map[:,:,0])

        return heat_map


    def update_params(self, heat_map):
        # preprocess & get infos
        params = [self.open, self.close, self.dhot]
        heuristics = self.get_heuristics(heat_map)
        state = params + heuristics
        if self.prev_state is None:
            self.prev_state = state

        # input to policy
        full_state = np.array(state + self.prev_state + self.memory)

        # output
        offsets = self.policy(full_state)

        # update
        self.prev_state = state
        self.open  += offsets[0]
        self.close += offsets[1]
        self.dhot  += offsets[2]
        self.memory[0] += 1             # frame counter
        self.memory[1] += offsets[3]    # learned param


    def policy(self, state):
        res = (0,0,0,0)
        if self.ask:
            # manual learner
            print('\nstate:\n',state)
            inp = input('new params eg "0,0,0,0"')
            inp = inp if len(inp)>3 else "0,0,0,0"
            res = eval(inp)
            self.ask=False

        return res
        
def get_heuristics(heat_map):
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(heat_map, 5, cv2.CV_32S)
    areas = stats[1:,-1]
    a_st = areas.std()
    a_me = areas.mean()
    n_wh = (heat_map>0).sum()
    return [n_cc, a_st, a_me, n_wh]
