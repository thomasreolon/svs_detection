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
    def __init__(self, d_close=1, d_open=3, d_hot=5, svs_ker=0, policy='', verbose=True, train=False):
        # Algorithm parameters
        self.kernels = self.get_kernels()
        self.verbose  = verbose
        self.er_k  = svs_ker
        self.open  = d_open
        self.close = d_close
        self.dhot  = d_hot
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
        self.heuristics = np.zeros(10)
        if hasattr(self.pred_reward, 'reset'):
            self.pred_reward.reset()

    def __call__(self, frame):
        # process as usual
        motion_map = super().__call__(frame)

        # update params
        self.update_params(motion_map[...,0])
        self.count += 1

        return motion_map

    def get_stateactions(self, motion_map):
        # preprocess & get infos
        params = np.array([self.close, self.open, self.dhot, self.er_k],  dtype=np.float32)
        s = 0.65 if self.training else 0.9
        self.heuristics = self.heuristics*s + get_heuristics(motion_map)*(1-s)
        state = np.concatenate((params, self.heuristics / (1-s**(self.count+1))  ))

        # get options
        actions = self.get_actions()
        state_actions = [np.nan_to_num(np.concatenate((a+state[:4],state[4:])), False, 0,1e2,-1e2) for a in actions]
        return state_actions

    def update_params(self, motion_map):
        state_actions = self.get_stateactions(motion_map)

        # pick best action
        tensors = [self.pred_reward(sa) for sa in state_actions]
        i = tensors.index(max(tensors))
        new_state = state_actions[i].astype(int) # new state

        if self.verbose and i>0:
            print(f'switching to: {(new_state[:4]).tolist()}')

        # update
        self.close = new_state[0]
        self.open  = new_state[1]
        self.dhot  = new_state[2]
        self.er_k  = new_state[3]
    
    def get_actions(self):
        ac = [(0,0,0,0)]
        if self.count<15:
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
            for a in [ # quicker convergence at inference
                (       0,          0,          0,      5-self.er_k), # high kernel
                (       0,          0,          0,      2-self.er_k), # mid kernel
                (       0,          0,          0,      0-self.er_k), # low kernel
                (      0,          0,20-self.open,      0),
                (      0,          0,10-self.open,      0),
                (      0,          0, 3-self.open,      0),
                (      0,9-self.open,10-self.open,      0),
                (      0,2-self.open,          0,      0),
                (1-self.close,     0,          0,      0),
                (      0,          4,          4,      0),
                (      0,          0,          4,      0),
                ((      0,         -4,         -4,      0) if self.open>self.close+4  else (1-self.close, 2-self.open,3-self.dhot,0)),
                ((      0,         -4,          0,      0) if self.open>self.close+4  else (1-self.close, 2-self.open,0,0)),
                ]:
                if not self.training and a not in ac: ac.append(a)
        return ac


def get_heuristics(motion_map):
    n_wh = (motion_map>0).sum() +1
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(motion_map, 5, cv2.CV_32S)
    if n_cc==1:
        a_st = 0
        a_me = 0
    else:
        areas = stats[1:,-1]
        a_st = areas.std()
        a_me = areas.mean()
    _,w,*_ = motion_map.shape
    n_wl = motion_map[:, :w//2].sum()
    n_wr = motion_map[:, w//2:].sum()
    m = cv2.moments(motion_map[:,:])
    m_y  = m['m10'] / n_wh
    m_x  = m['m01'] / n_wh
    d_y  = m['mu20']**0.5 / n_wh
    d_x  = m['mu02']**0.5 / n_wh

    #           n_blobs,  blob_sizes, blob_sizes, white,    left,   right,      center,   distance from center
    return np.array([n_cc/100, a_st/100,  a_me/40,    n_wh/1e3, n_wl/5e3, n_wr/5e3, m_x/1e4, m_y/1e4, d_x/10, d_y/10])
