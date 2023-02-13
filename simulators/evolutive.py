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
        self.open  = d_open
        self.close = d_close
        self.dhot  = d_hot
        self.updateevery = updateevery
        self.pred_reward = torch.nn.Sequential(
            torch.nn.Linear(15, 5),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(5, 1),
        )
        self.pred_reward.eval()

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        self.prev_state = None
        self.count = -self.updateevery*2

    def __call__(self, frame):
        # process as usual
        heat_map = super().__call__(frame)

        # update params
        if self.count % self.updateevery == 0:
            self.update_params(heat_map[...,0])
        self.count += 1

        return heat_map


    def update_params(self, heat_map):
        # preprocess & get infos
        params = [self.close, self.open, self.dhot]
        heuristics = get_heuristics(heat_map)
        state = params + heuristics
        if self.prev_state is None:
            self.prev_state = state

        # input to policy
        full_state = np.array(state + self.prev_state + [self.count])

        # output
        action = self.policy(full_state)

        # update
        self.prev_state = state
        self.close += action[0]
        self.open  += action[1]
        self.dhot  += action[2]

    def policy(self, state):
        actions = self.get_actions()
        print(actions)
        tensors = [self.score(state,a) for a in actions]

        scores = np.exp(np.array([ x.detach().item() for x in tensors]))
        print(scores)
        scores = scores/scores.sum()
        print(scores)
        i = np.random.choice(list(range(len(actions))), p=scores)
        self._pred = tensors[i]
        print(actions[i], '\n_______________________________')
        return actions[i]

    def score(self, state, action):
        state[:3] += action
        tmp = torch.is_grad_enabled()
        torch.set_grad_enabled(self.pred_reward.training)
        pred_reward =  self.pred_reward(torch.tensor(state,dtype=torch.float)[None])
        torch.set_grad_enabled(tmp)
        pred_reward = 10/(1 + (-pred_reward).exp()/100 )
        return pred_reward

    def _init_weight(self):
        p = pathlib.Path(__file__).parent.resolve().__str__()
        p = p+'/../policy.pt'
        if os.path.exists(p):
            self.pred_reward.load_state_dict(torch.load(p))

    def get_actions(self):
        ac = [(0,0,0)]
        if self.count>=0:
            ac = ac + [
                (1-self.close, 3-self.open, 4-self.dhot),
                (3-self.close, 2-self.open, 4-self.dhot),
                (3-self.close, 4-self.open, 10-self.dhot),
            ]
            if self.dhot<20:
                ac.append((0,0,1))
            if self.dhot-1>max(self.open, self.close):
                ac.append((0,0,-1))
            if self.open+1<self.dhot:
                ac.append((0,1,0))
            if self.open>1:
                ac.append((0,-1,0))
            if self.close+1<self.dhot:
                ac.append((1,0,0))
            if self.close>1:
                ac.append((-1,0,0))
        return ac



def get_heuristics(heat_map):
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(heat_map, 5, cv2.CV_32S)
    areas = stats[1:,-1]
    a_st = np.nan_to_num(areas.std(), nan=0)
    a_me = np.nan_to_num(areas.mean(), nan=0)
    n_wh = (heat_map>0).sum()
    return [n_cc, a_st, a_me, n_wh]
