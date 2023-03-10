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
from sklearn.ensemble import IsolationForest

############################### POLICY v3-4 ###############################

def clean_h(row):
  heuristics = eval(row[1]['heuristics'].replace('nan', 'np.nan'))
  return np.nan_to_num(np.array(heuristics), False, 0)
def clean_p(row):
  return np.array(eval(row[1]['params'])[1])
def clean_p2(row):
  map_, p1 = eval(row[1]['params'])
  p2 = [a[1] for a in eval(row[1]['other']) if a[0]+0.05>map_]
  return np.array([p1, p1] + p2 if len(p2) else [p1, p1])
def dummy(x): return [np.array([0])]
class FixPredictorV2():
  def __init__(self, data, n_clust=3) -> None:
    # preprocess data
    data['best'] = list(map(clean_p, data.iterrows()))
    data['top'] = list(map(clean_p2, data.iterrows()))
    data['he'] = list(map(clean_h, data.iterrows()))

    # M*heuristics, M*target_parameters, N*parameters
    x,y,p = self.make_dataset(data, n_clust)
    clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
    if n_clust>1:
      clf.fit(x, y)
    else:
      clf.decision_function = dummy # don't need classifier if just 1 class
    self.clf = clf
    self.params = p
    self.prev_p = [np.zeros(len(p)), 0, np.zeros(4)]
    print('targets:\n',p)
  
  def reset(self):
    self.prev_p[0] *= 0
    self.prev_p[1]  = 0
    self.prev_p[2] *= 0

  def __call__(self, stateaction):
    # update probability to belong to each cluster
    x = np.nan_to_num(stateaction[4:], False, 0) # heuristics
    prob_v = np.exp(self.clf.decision_function([x])[0]) # get video probabilities
    p = (prob_v / prob_v.sum())   # probability to belong to cluster 
    self.prev_p[0] += p           # update means pt.1
    self.prev_p[1] += 1           # update means pt.2

    # new target = weighted sum of clustered parameters
    prob_v = self.prev_p[0] / self.prev_p[1] # average prob through time
    target = (self.params * prob_v.reshape(-1,1)).sum(axis=0) # weighted sum

    # # velocity
    # future = (self.params * p.reshape(-1,1)).sum(axis=0)
    # self.prev_p[2] = 0.75*self.prev_p[2] + 0.25*(future - target)
    # target += self.prev_p[2]

    return 32 - np.abs(stateaction[:4]-target).sum() # the closer to target the highest the score

  def make_dataset(self, data, n_clust):
    x,y,p = [], [], []
    used = set()
    # get top configurations
    for _, row in data.iterrows():
      p += [a for a in row['top']]

    # remove outliers
    p = np.array(p)
    inliers = IsolationForest(contamination=0.15, max_samples=100, random_state=0).fit_predict(p)
    p = p[inliers>0]

    # find N clusters of configurations
    kmeans  = KMeans(n_clusters=n_clust, random_state=1, n_init="auto").fit(p)
    new_p = kmeans.cluster_centers_.copy()
    for _, row in data.iterrows():
      i = kmeans.predict([row['best']])[0]
      x += [a for a in row['he']]
      y += [i] * len(row['he'])
      used.add(i)
    
    # assure there ara as many classes in the SMV as there are clusters
    for i in range(n_clust):
      if i not in used:
        x.append(x[-i-1]*-1)
        y.append(i)
    return x,y,new_p


def clean_m(row):
  return np.array(eval(row[1]['params'])[0])
class FixPredictorV1():
  def __init__(self, data) -> None:
    data['param'] = list(map(clean_p, data.iterrows()))
    data['map'] = list(map(clean_m, data.iterrows()))
    worst = data[data['map']==data['map'].min()].iloc[0]
    self.target = np.array(worst['param'])
  def __call__(self, stateaction):
    return 32 - np.abs(stateaction[:4]-self.target).sum() # the closer to target the highest the score


def clean_h2(row):
  heuristics = eval(row[1]['heuristics'].replace('array', '').replace('nan', 'np.nan'))
  return np.nan_to_num(np.array(heuristics), False, 0)
def clean_p3(row):
  return np.array(eval(row[1]['params']))
class NNPredictorV1():
  sizes = ['ex','xs','s','m','l','xl']
  def __init__(self, data, size='s') -> None:
    data['par'] = list(map(clean_p3, data.iterrows()))
    data['he'] = list(map(clean_h2, data.iterrows()))

    print('training NN...')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_,y_ = self.make_dataset(data)
    net = self.make_nn(size).to(dev)
    self.net = net

    # pretrain # learn (state,action) --> reward
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.8, 0.9))
    for _ in range(3001):
      x,y  = self.sample(x_,y_,dev)
      yp = net(x)
      loss = ((y-yp)**2).mean()
      loss.backward()
      opt.step()
      opt.zero_grad()
      if _ == 40: opt = opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.8, 0.99999))
      if _ == 2000: opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    # finetune as RL # learn (state,action) --> (reward + 0.24*future_reward)/1.24 (should smooth the prediction)
    lr = 6e-6 + min(1e-2, 10/sum(p.numel() for p in net.parameters())**2)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.95, 0.999))
    for _ in range(1000):
      x,y  = self.sample(x_,y_,dev,16)
      for i, xi in enumerate(x):
        fr = self.future_reward(xi)
        y[i] += -0.2*y[i] + 0.2*fr
      yp = net(x)
      loss = ((y-yp)**2).mean()
      loss.backward()
      opt.step()
      opt.zero_grad()
      if _ == 900: opt = torch.optim.SGD(net.parameters(), lr=6e-6)
    
    self.net = net.cpu().eval()

  @torch.no_grad()
  def __call__(self, x):
    if isinstance(x, np.ndarray):
      x = torch.tensor(x)[None].float()
    return self.net(x).item()

  def sample(self, x_, y_, dev, bs=256):
    idxs = (len(x_) * np.random.rand(int(bs*0.9))).astype(int).tolist()
    x,y  = x_[idxs].view(-1,14), y_[idxs].view(-1,1)
    x2,y2 = self.bad_samples(bs-len(idxs))
    return torch.cat((x,x2),0).to(dev), torch.cat((y,y2),0).to(dev)

  def bad_samples(self, n):
    """we want our network to predict bad scores for configurations that makes no sense"""
    y = torch.tensor([-2]*n)[:,None]
    x = (torch.rand(n,14)-0.05)*30
    x[:,1] *= 1.3
    x[:,2] *= 3
    x[:,4:] /= 10
    bad = (x - torch.tensor([[5,5,5,2,1,1,1,1,1,1,1,1,1,1]])).abs().sum(dim=1) > 30
    x[~bad, 2] += 100
    return x,y

  def make_dataset(self, data):
    x_ = [] ; y_ = []
    for _, row in data.iterrows():
      for a in row['he']:
        x_.append(torch.from_numpy(np.concatenate((row['par'], a))))
        y_.append(torch.tensor([(row['map']-0.2)*10]))
    return torch.stack(x_).float(),torch.stack(y_).float()
  
  def future_reward(self, x:torch.Tensor):
    best = -1e99
    self.net.eval()
    for a in [(0,0,0,0),(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),(0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
      tmp = x.clone()
      tmp[:4] += torch.tensor(list(a), device=tmp.device, dtype=tmp.dtype)
      future_r = self(tmp[None])
      if future_r > best:
        best = future_r
    self.net.train()
    return best

  def make_nn(self, size='s', n_in=14):
    if size=='xs':
      net = nn.Sequential(
        nn.BatchNorm1d(n_in),
        nn.Linear(n_in, 4),
        nn.ReLU(),
        nn.Linear(4,1),
      )
    elif size=='s':
      net = nn.Sequential(
        nn.BatchNorm1d(n_in),
        nn.Linear(n_in, 4),
        nn.ReLU(),
        nn.Linear(4,4),
        nn.Dropout(0.05),
        nn.ReLU(),
        nn.Linear(4,4),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(4,1),
      )
    elif size=='m':
      net = nn.Sequential(
        nn.BatchNorm1d(n_in),
        nn.Linear(n_in, 32),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(32,1),
      )
    elif size=='l':
      net = nn.Sequential(
        nn.BatchNorm1d(n_in),
        nn.Linear(n_in, 16),
        nn.ReLU(),
        nn.Linear(16,32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32,4),
        nn.ReLU(),
        nn.Linear(4,1),
      )
    elif size=='xl':
      net = nn.Sequential(
        nn.BatchNorm1d(n_in),
        nn.Linear(n_in, 256),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(256, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
        nn.ReLU(),
        nn.Linear(2,1),
      )
    elif size=='ex':
      net = ExperimNN(n_in)
    return net.float()

class ExperimNN(nn.Module):
  def __init__(self, n_in) -> None:
    super().__init__()
    self.lin1 = nn.Linear(n_in, 16)
    self.lin2 = nn.Linear(32, 4)
    self.lin3 = nn.Linear(4, 1)
    self.act  = nn.ReLU(True)
    self.t_b  = nn.Parameter(torch.ones(1,4,4)*3+torch.rand(1,4,4)*2, True)
    self.t_w  = nn.Parameter(torch.rand(1,4,4), True)
  def forward(self, x):
    params = x[:, :4, None]
    dist = (params - self.t_b)**2 * self.t_w
    y = self.lin1(x)
    y = torch.cat((y, dist.flatten(1)), dim=1)
    return self.lin3(self.act(self.lin2(y)))
    


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


