import numpy as np
import cv2
import torch
import os

from .forensor_sim import StaticSVS

"""
Dynamic Simulator
perameters change with a previously learned policy
"""

class RLearnSVS(StaticSVS):
    name = 'policy'
    def __init__(self, d_close=1, d_open=3, d_hot=5, policy='', updateevery=1, verbose=True, train=False):
        # Algorithm parameters
        self.kernels = self.get_kernels()
        self.verbose  = verbose
        self.er_k  = 0
        self.open  = d_open
        self.close = d_close
        self.dhot  = d_hot
        self.updateevery = updateevery
        self.training = train
        self.pred_reward = self._load_policy(policy)

    def _load_policy(self, policy_weights):
        policy = lambda x: 0
        if os.path.isfile(policy_weights):
            print(f'loaded policy: {policy_weights[-20:]}')
            policy = torch.load(policy_weights)
        return policy

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        # self.prev_state = None
        self.count = 0

    def __call__(self, frame):
        # process as usual
        motion_map = super().__call__(frame)

        # update params
        if self.count % self.updateevery == 0:
            self.update_params(motion_map[...,0])
        self.count += 1

        return motion_map

    def get_stateactions(self, motion_map):
        # preprocess & get infos
        params = [self.close, self.open, self.dhot, self.er_k]
        heuristics = get_heuristics(motion_map)
        state = np.array(params + heuristics)

        # get options
        actions = self.get_actions()
        state_actions = [np.nan_to_num(np.concatenate((a+state[:4],state)), False, 0,1e2,-1e2) for a in actions]
        return state_actions

    def update_params(self, motion_map):
        state_actions = self.get_stateactions(motion_map)

        # pick best action
        tensors = [self.pred_reward(sa.astype(float)) for sa in state_actions]
        i = tensors.index(max(tensors))
        new_state = state_actions[i].astype(int) # new state

        if self.verbose and i>0:
            print(f'switching: {new_state[4:8].tolist()} --> {(new_state[:4]).tolist()}')

        # update
        self.close = new_state[0]
        self.open  = new_state[1]
        self.dhot  = new_state[2]
        self.er_k  = new_state[3]
    
    def get_actions(self):
        ac = [(0,0,0,0)]
        if self.count>=0:
            if self.dhot<30:
                ac.append((0,0,1,0))
            if self.dhot-1>max(self.open, self.close):
                ac.append((0,0,-1,0))
            if self.open+1<self.dhot:
                ac.append((0,1,0,0))
            if self.open>self.close+1:  #1:  # allows 1,1,2
                ac.append((0,-1,0,0))
            if self.close+1<self.open: #dhot: # allows 1,1,2
                ac.append((1,0,0,0))
            if self.close>1:
                ac.append((-1,0,0,0))
            if self.er_k>0:
                ac.append((0,0,0,-1))
            if self.er_k+1<len(self.kernels):
                ac.append((0,0,0,1))
            for a in [ # faster exploration (a little more unstable)
                (       0,          0,          0,      5-self.er_k), # high kernel
                (       0,          0,          0,      2-self.er_k), # mid kernel
                (       0,          0,          0,      0-self.er_k), # low kernel
                ((      0,          5,          5,      0) if self.dhot<15            else (1-self.close, 10-self.open,11-self.dhot,0)),
                ((      0,          0,          5,      0) if self.dhot<15            else (1-self.close, 2-self.open,11-self.dhot,0)),
                ((      0,         -5,         -5,      0) if self.open>self.close+5  else (1-self.close, 2-self.open,3-self.dhot,0)),
                ((      0,         -5,          0,      0) if self.open>self.close+5  else (1-self.close, 2-self.open,11-self.dhot,0)),
                ]:
                if a not in ac: ac.append(a)
        return ac


def get_heuristics(motion_map):
    n_wh = (motion_map>0).sum()
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(motion_map, 5, cv2.CV_32S)
    if n_cc==1:
        a_st = 0
        a_me = 0
    else:
        areas = stats[1:,-1]
        a_st = areas.std()
        a_me = areas.mean()
    
    return [n_cc/100, a_st, a_me/10, n_wh/1000]
