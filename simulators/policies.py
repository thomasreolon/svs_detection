import torch, torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats as st
from collections import defaultdict

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans

############################### POLICY v1 ###############################




def clean_h(row):
  heuristics = eval(row[1]['heuristics'].replace('nan', 'np.nan'))
  return np.nan_to_num(np.array(heuristics), False, 0)
def clean_p(row):
  return np.array(eval(row[1]['params'])[1])
class FixPredictorV2():
  def __init__(self, data) -> None:
    data['best'] = list(map(clean_p, data.iterrows()))
    data['he'] = list(map(clean_h, data.iterrows()))

    x,y,p = self.make_dataset(data)
    clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
    print('training svm...')
    clf.fit(x, y)
    self.clf = clf
    self.params = p
    print('targets:\n',p)
  
  def __call__(self, stateaction):
    x = np.nan_to_num(stateaction[4:], False, 0)
    i = self.clf.predict([x])[0]
    p = self.params[i]
    return 32 - np.abs(stateaction[:4]-p).sum()

  def make_dataset(self, data):
    x,y,p = [], [], []
    for _, row in data.iterrows():
      p.append(row['best'])

    print('training kmeans...')
    kmeans  = KMeans(n_clusters=3, random_state=1, n_init="auto").fit(p)
    new_p = kmeans.cluster_centers_.copy()

    for _, row in data.iterrows():
      i = kmeans.predict([row['best']])[0]
      x += [a for a in row['he']]
      y += [i] * len(row['he'])

    return x,y,new_p


class FixPredictorV1():
  def __init__(self, data) -> None:
    data['best'] = list(map(clean_p, data.iterrows()))
    data['he'] = list(map(clean_h, data.iterrows()))

    x,y,p = self.make_dataset(data)
    clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
    clf.fit(x, y)
    self.clf = clf
    self.params = p
  
  def __call__(self, stateaction):
    x = np.nan_to_num(stateaction[4:], False, 0)
    i = self.clf.predict([x])[0]
    p = self.params[i]
    return 32 - np.abs(stateaction[:4]-p).sum()

  def make_dataset(self, data):
    x,y,p = [], [], []
    for i, row in data.iterrows():
      x += [a for a in row['he'][::100]]
      y += [i] * len(row['he'][::100])
      p.append(row['best'])
    return x,y,p

############################### POLICY v2 ############################### 

class SVMPredictor():
  def __init__(self, data):
    X = np.stack([y for y in data['state_action']])
    y = np.stack([y for y in data['map']])
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X, y)
    self.clf = regr

  def __call__(self, stateaction):
    # update video knowledge
    return self.clf.predict(stateaction[None])

class NNPredictor():
  def __init__(self, data, hid_layer_size=16, reward=(0,0,1)):
    self.reward = torch.tensor(reward) / sum(reward)
    # preproc data
    data['vid_class'] = list(map(lambda x: x.split(':')[-2], data['video']))
    cc = data.groupby('vid_class').count()
    counts = {k:len(data)/5/v for k,v in zip(list(cc.index), cc.values[:,0])}
    data['map_reward'] = [0]*len(data)
    base_maps = {}
    for i, row in data.iterrows():
      base = self.get_base_map(data, row['idx'], base_maps)
      data.loc[i,'map_reward'] = (data.loc[i,'map'] - base)
    # preproc normalize
    means = data['map_reward'].mean(), data['map'].mean(), data['reward'].mean()
    stds = data['map_reward'].std(), data['map'].std(), np.nan_to_num(data['reward'].std(), 0, 1.)
    n = (means, stds)
    data['map_reward']  = (data['map_reward'] - n[0][0]) / (n[1][1]+1e-5) # increase decrease of map wrt neighbours
    data['reward']      = (data['reward']     - n[0][2]) / (n[1][2]+1e-5) # increase decrease of yolo detection loss wrt neighbours
    data['map']         = (data['map']        - n[0][1]) / (n[1][1]+1e-5) # map for that configuration
    
    # create NN
    hls = hid_layer_size
    net = nn.Sequential(
        nn.BatchNorm1d(12),
        nn.Linear(12, hls),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(hls,3),
    )

    # pretrain
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.8, 0.9))
    for _ in range(5001):
        # learn (state,action) --> reward
        batch = data.sample(256)
        x = torch.from_numpy(np.stack([x for x in batch['state_action']])).float()
        y1 = torch.from_numpy(np.stack([x for x in batch['map_reward']])).float()
        y2 = torch.from_numpy(np.stack([x for x in batch['reward']])).float()
        y3 = torch.from_numpy(np.stack([x for x in batch['map']])).float()
        y = torch.stack((y1,y2,y3), dim=1)
        yp = net(x)
        loss = ((y-yp)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if _ == 20: opt = opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.8, 0.99999))
        if _ == 4000: opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    self.net = net

    # finetune as RL
    opt = torch.optim.Adam(net.parameters(), lr=6e-6, betas=(0.95, 0.999))
    discount = 0.1666
    for e in range(1000):
        xs,ys,rs = [], [], []
        for _ in range(16):
          # learn (state,action) --> (reward + 0.2*future_reward)/1.2
          batch = data.sample(1)
          reward = torch.tensor([float(batch['map_reward']),float(batch['reward']),float(batch['map'])])
          future_r = self.get_best_future(batch['state_action'])

          r = counts[batch['vid_class'].iloc[0]]
          y = (reward*(1-discount) + future_r*discount).float()
          x = torch.from_numpy(batch['state_action'].iloc[0]).float()
          xs.append(x) ; ys.append(y) ; rs.append(r)

        yp = net(torch.stack(xs))
        loss = ((torch.stack(ys)-yp)**2 * torch.tensor(rs)[:,None]).mean(dim=0)
        loss.sum().backward()
        opt.step()
        opt.zero_grad()
    net.eval()

  @torch.no_grad()
  def get_best_future(self, stateaction):
    best = -1e99, None
    stateaction[4:8] = stateaction[:4]
    self.net.eval()
    for a in [(0,0,0,0),(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),(0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
      new = np.array(stateaction)[0].copy()
      new[:4] += a
      pred = self.net(torch.tensor(new)[None].float())[0]
      score = (pred*self.reward).sum().item()
      if score > best[0]:
        best = score, pred
    self.net.train()
    return best[1]

  def get_base_map(self, data, idx, base_map):
    if idx not in base_map:
      tmp = data[data['idx']==idx]
      map_ = tmp.iloc[0]['map']
      base_map[idx] = map_
    return base_map[idx]

  @torch.no_grad()
  def __call__(self, stateaction):
    pred = self.net(torch.tensor(stateaction)[None].float())
    score = (pred[0]*self.reward).sum().item()
    return score

def get_mode(sa):
  res = []
  if isinstance(sa, pd.DataFrame):
    sa = sa['state_action']
  for i in range(4):
    tmp = np.array(list(map(lambda x:x[i], sa)))
    res.append(st.mode(tmp)[0])
  return np.concatenate(res)
class FixPredictor():
  def __init__(self, data, mode=0):

    # best params per video
    if mode==0:
      data['vid_class']  = list(map(lambda x: x.split(':')[-2], data['video']))               # dataset&data augmentation
      best = data.groupby('video')['map'].max()
      means = {'vid_class':[], 'params':[]}
      for k in best.keys():
        tmp = data[data['video']==k]
        tmp = tmp[tmp['map']==best[k]]
        means['vid_class'].append(tmp.iloc[0]['vid_class'])
        means['params'].append(get_mode(tmp))
      means  = pd.DataFrame.from_dict(means)
      best = means.groupby('vid_class')['params'].mean()
      self.best = np.stack(best.values).reshape(-1,4)
    elif mode==1:
      data['vid_class']  = list(map(lambda x: x.split(':')[-2].split('-')[0], data['video']))  
      vid_stat = defaultdict(lambda: {'params':[], 'map':[]})
      for i, row in data.iterrows():
        v = row['video'].split(':')[0]
        s = row['state_action'][:4]
        m = row['map'] * 10
        vid_stat[v]['params'].append(s)
        vid_stat[v]['map'].append(m)
      best = {k:[] for k in data['vid_class'].unique()}
      for v, stats in vid_stat.items():
        m_prob = np.exp(np.array(stats['map']))
        m_prob = (m_prob / m_prob.sum()).reshape(-1,1)
        params = np.stack(stats['params']).reshape(-1,4)
        params = (params*m_prob).sum(axis=0)
        if 'synth' in v:
          best['synth'].append(params)
        if 'MOT17' in v:
          best['mot17'].append(params)
        if 'vid_' in v:
          best['streets23'].append(params)
      best = {k:get_mode(v) for k,v in best.items()}
      self.best = np.stack(list(best.values())).reshape(-1,4)
    elif mode==2:
      data['vid_class']  = list(map(lambda x: x.split(':')[-2].split('-')[0], data['video']))
      best = {k:[] for k in data['vid_class'].unique()}
      for idx in data.idx.unique():
        tmp = data[data['idx']==idx]
        bestmap = tmp['map'].max()
        row_base = tmp.iloc[0]
        if row_base['map'] + 0.01 > bestmap:
          best[ row_base['vid_class'] ].append(row_base['state_action'])
      best = {k:get_mode(v) for k,v in best.items()}
      self.best = np.stack(list(best.values())).reshape(-1,4)

    # recognizing the video
    X = list(map(lambda x:x[8:12], data['state_action']))
    vid = {k:i for i,k in enumerate(best.keys())}
    y = list(map(lambda x: vid[x], data['vid_class']))
    clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
    clf.fit(X, y)
    self.vid_clf = clf
    self.prev_p = [np.zeros(len(best.keys())), 0]

  def __call__(self, stateaction):
    # update video knowledge
    self.prev_p[1] += 1
    prob_v = np.exp(self.vid_clf.decision_function([stateaction[-4:]])[0])
    self.prev_p[0] += (prob_v / prob_v.sum())

    # target config
    prob_v = self.prev_p[0] / self.prev_p[1]
    target = (self.best * prob_v.reshape(-1,1)).sum(axis=0)

    # score
    return 100 - ((stateaction[:4] - target)**2).sum() /3


