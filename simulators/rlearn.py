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

def get_policy_type(type='neural_net'):
    # maybe will support probabilistic models in the future
    if type == 'neural_net':
        return torch.nn.Sequential(
            torch.nn.Linear(17, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )


class RLearnSVS(StaticSVS):
    name = 'policy'
    def __init__(self, d_close=1, d_open=3, d_hot=5, policy='policy.pt', updateevery=3, verbose=True):
        # Algorithm parameters
        self.kernels = self.get_kernels()
        self.verbose  = verbose
        self.er_k  = 0
        self.open  = d_open
        self.close = d_close
        self.dhot  = d_hot
        self.updateevery = updateevery
        self.pred_reward = get_policy_type()
        self._init_weight(policy)
        self.pred_reward.eval()
        self._i = -1

    def init_video(self, init_threshold, std):
        """initial values for the threshold (background image without any moving object)"""
        super().init_video(init_threshold, std)
        self.prev_state = None
        self.count = -self.updateevery*2

    def __call__(self, frame):
        # process as usual
        motion_map = super().__call__(frame)

        # update params
        if self.count % self.updateevery == 0:
            self.update_params(motion_map[...,0])
        self.count += 1

        return motion_map

    def update_params(self, motion_map):
        # preprocess & get infos
        params = [self.close, self.open, self.dhot, self.er_k]
        heuristics = get_heuristics(motion_map)
        state = params + heuristics
        if self.prev_state is None:
            self.prev_state = state

        # input to policy
        full_state = np.array(state + self.prev_state + [min(50,max(0,self.count))])

        # output
        action = self.policy(full_state)

        # update
        self.prev_state = state
        self.close += action[0]
        self.open  += action[1]
        self.dhot  += action[2]
        self.er_k  += action[3]

    def policy(self, state):
        actions = self.get_actions()

        tensors = [self.score(state,a) for a in actions]

        if self.pred_reward.training:
            scores = np.array([ x.detach().item() for x in tensors])  
            scores = scores * (20+max(0,min(1000,self.count))) / 20  # Exploitation through time
            scores = np.exp(scores)
            scores = scores/scores.sum()
            i = np.random.choice(list(range(len(actions))), p=scores)
            i = (i+1) % len(actions) if i==self._i else i
            self._pred = tensors[i] ; self._i = i
        else:
            i = tensors.index(max(tensors))

        if self.verbose and i>0:
            print(f'switching: {state[:4].tolist()} --> {(state[:4]+actions[i]).tolist()}')

        return actions[i]

    def score(self, state, action):
        state = state.copy()
        state[:4] += action
        state = np.nan_to_num(state, False, 0,1e5,-1e5)
        tmp = torch.is_grad_enabled()
        torch.set_grad_enabled(self.pred_reward.training)
        pred_reward =  self.pred_reward(torch.tensor(state,dtype=torch.float)[None])
        torch.set_grad_enabled(tmp)
        return pred_reward

    def _init_weight(self, policy_weights):
        # policy_weights: absolute path or relative path wrt. home
        p = policy_weights if os.path.exists(policy_weights) else f'{pathlib.Path(__file__).parent.resolve().__str__()}/../{policy_weights}'
        if os.path.exists(p):
            print(f'loaded policy: {p[-20:]}')
            self.pred_reward.load_state_dict(torch.load(p))

    def get_actions(self):
        ac = [(0,0,0,0)]
        if self.count>=0:
            if self.dhot<20:
                ac.append((0,0,1,0))
            if self.dhot-1>max(self.open, self.close):
                ac.append((0,0,-1,0))
            if self.open+1<self.dhot:
                ac.append((0,1,0,0))
            if self.open>self.close+1:  #1: # allows 1,1,2
                ac.append((0,-1,0,0))
            if self.close+1<self.dhot:
                ac.append((1,0,0,0))
            if self.close>1:
                ac.append((-1,0,0,0))
            if self.er_k>0:
                ac.append((0,0,0,-1))
            if self.er_k+1<len(self.kernels):
                ac.append((0,0,0,1))
            for a in [ # checkpoints
                (1-self.close, 3-self.open, 4-self.dhot, 0-self.er_k),
                (3-self.close, 2-self.open, 4-self.dhot, 2-self.er_k),
                (3-self.close, 4-self.open, 10-self.dhot, 3-self.er_k),]:
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
        a_st = np.nan_to_num(areas.std(), nan=0)
        a_me = np.nan_to_num(areas.mean(), nan=0)
    
    return [n_cc, a_st, a_me, n_wh]
