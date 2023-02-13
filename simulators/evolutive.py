import numpy as np
import cv2
import torch
import pathlib
import os

from .forensor_sim import StaticSVS

"""
Dynamic Simulator
perameters change with a previously learned policy
"""

class EvolvedSVS(StaticSVS):
    name = 'evolved'
    def __init__(self, d_close=1, d_open=3, d_hot=5, updateevery=3):
        # Algorithm parameters
        self.erosion_kernel = np.ones((3, 3), np.uint8)
        self.open = d_open
        self.close = d_close
        self.dhot = d_hot
        self.ask=False
        self.updateevery = updateevery
        self.pred_reward = torch.nn.Sequential(
            torch.nn.Linear(15, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        self.prev_state = None
        self.count = -self.updateevery


    def __call__(self, frame):
        # process as usual
        heat_map = super().__call__(frame)

        # update params
        if self.count>0 and self.count % self.updateevery == 0:
            self.update_params(heat_map[...,0])
        self.count += 1

        return heat_map


    def update_params(self, heat_map):
        # preprocess & get infos
        params = [self.open, self.close, self.dhot]
        heuristics = self.get_heuristics(heat_map)
        state = params + heuristics
        if self.prev_state is None:
            self.prev_state = state

        # input to policy
        full_state = np.array(state + self.prev_state + [self.count])

        # output
        action = self.policy(full_state)

        # update
        self.prev_state = state
        self.open  += action[0]
        self.close += action[1]
        self.dhot  += action[2]

    def policy(self, state):
        actions = self.get_actions()
        scores = np.exp(np.array([self.score(state,a) for a in actions]))
        scores = scores/scores.sum()
        action = np.random.choice(actions, p=scores)
        return action

    def score(self, state, action):
        state[:3] += action
        return self.pred_reward(torch.tensor(state)).item()

    def _init_weight(self):
        p = pathlib.Path(__file__).parent.resolve().__str__()
        p = p+'/../policy.pt'
        if os.path.exists(p):
            self.pred_reward.load_state_dict(torch.load(p))

    def get_actions(self):
        ac = [(0,0,0)]
        if self.dhot<20:
            ac.append(0,0,1)
        if self.dhot-1>max(self.open, self.close):
            ac.append(0,0,-1)
        if self.open+1<self.dhot:
            ac.append(0,1,0)
        if self.open>1:
            ac.append(0,-1,0)
        if self.close+1<self.dhot:
            ac.append(1,0,0)
        if self.close>1:
            ac.append(-1,0,0)



def get_heuristics(heat_map):
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(heat_map, 5, cv2.CV_32S)
    areas = stats[1:,-1]
    a_st = areas.std()
    a_me = areas.mean()
    n_wh = (heat_map>0).sum()
    return [n_cc, a_st, a_me, n_wh]
