import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..')
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

from configs.defaults import get_args_parser
from datasets.mot_svs_cache import FastDataset
import utils.debugging as D


args = get_args_parser().parse_args()
args.dataset = 'all'
args.crop_svs = True

dataset = FastDataset(args, True, False)

boxes = []

for i in tqdm(range(len(dataset))):
    _,_,y,_ = dataset[i]
    for box in y:
        w, h = (box[-2]*160).item(), (box[-1]*128).item()
        if w*h==0:continue
        boxes.append([w,h])

boxes = np.array(boxes)

print('-- K=4')
kmeans  = KMeans(n_clusters=4, random_state=1, n_init="auto").fit(boxes)
anchors = kmeans.cluster_centers_.copy()
anchors = sorted(anchors, key=lambda xy:xy[0]*xy[1])
print(np.array(anchors).round().astype(int))

"""
[[ 3  8]
 [ 7 18]
 [15 36]
 [33 74]]
"""

print('-- K=9')
kmeans  = KMeans(n_clusters=9, random_state=1, n_init="auto").fit(boxes)
anchors = kmeans.cluster_centers_.copy()
anchors = sorted(anchors, key=lambda xy:xy[0]*xy[1])
print(np.array(anchors).round().astype(int))

"""
[[ 3  6]
 [ 5 13]
 [ 7 18]
 [10 25]
 [14 33]
 [17 47]
 [30 54]
 [24 70]
 [40 85]]
"""

""" w cars
[[  4   7]
 [  6  14]
 [ 14  13]
 [  9  22]
 [ 13  32]
 [ 31  29]
 [ 21  50]
 [ 32  80]
 [ 55 111]]
"""

print('-- K=6')
kmeans  = KMeans(n_clusters=6, random_state=1, n_init="auto").fit(boxes)
anchors = kmeans.cluster_centers_.copy()
anchors = sorted(anchors, key=lambda xy:xy[0]*xy[1])
print(np.array(anchors).round().astype(int))
"""
[[ 3  7]
 [ 5 14]
 [ 8 20]
 [13 31]
 [20 50]
 [36 80]]
"""
