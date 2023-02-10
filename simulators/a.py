import cv2
import numpy as np


heat_map = np.zeros((100,100))
heat_map[0:52,0: ] = 255
heat_map[30:40,20:50 ] = 255
heat_map[20:40,30:40 ] = 255

heat_map[52:56,33:99 ] = 255

heat_map[70:72,3:95 ] = 255

heat_map[88:89,88:89 ] = 255

heat_map = np.uint8(heat_map)
output = cv2.connectedComponentsWithStats(
    heat_map, 5, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
print('\n-----------------\n'.join([f'\n{(x.shape if not isinstance(x,int) else x)}\n{x}' for x in output]))


n_cc = numLabels
areas = stats[1:,-1]
a_st = areas.std()
a_me = areas.mean()
n_wh = (heat_map>0).sum()

print('/////////////////////////////////\n')
print(n_cc, a_st, a_me, n_wh)


cv2.imshow('h', heat_map)
cv2.waitKey()