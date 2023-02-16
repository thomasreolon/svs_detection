from ._loss import ComputeLoss
from .mlp import MLPDetector
from .mlp_v2 import MLPDetectorv2
from .yolov5 import YoloNet5, YoloNet8, YoloNetPhi

def build(model_name, ch_in = 1):
    if model_name == 'mlp1':
        return MLPDetector(ch_in)
    if model_name == 'mlp2':
        return MLPDetectorv2(ch_in)
    if model_name == 'yolo5':
        return YoloNet5(ch_in)
    if model_name == 'yolo8':
        return YoloNet8(ch_in)
    if model_name == 'yolophi':
        return YoloNetPhi(ch_in)
    if model_name == 'opt_yolophi':
        return build_special(['yolophi', 0.15000000000000002, 0.1, 0.05, 8, 0.82, False, True, [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]])
    # if model_name == 'opt_mlp2':
    #     return build_special(['mlp2', False, 2.2, 6, 3, [[1, 4, 4, 8, 16, 16, 2, 2], [44, 44, 10, 30, 50, 70, 60, 110]]])
    # if model_name == 'opt_yolo8':
    #     return build_special(['yolo8', 0.33, 0.5])
    if model_name == 'opt_phi1':
        return build_special(['yolophi',  0.2,  0.55,  0.05,  9,  1.12,  False,  True,  [[10, 13, 16, 30, 33, 23],   [30, 61, 17, 31, 59, 119],   [116, 90, 156, 198, 60, 110]]])
    if model_name == 'opt_phi2':
        return build_special(['yolophi',  0.1,  0.65,  0.05,  7,  1.12,  True,  True,  [[10, 13, 16, 30, 33, 23],   [30, 61, 17, 31, 59, 119],   [116, 90, 156, 198, 60, 110]]])
    if model_name == 'opt_y5':
        return build_special(['yolo5',  0.15,  0.52,  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]])
    if get_num(model_name):
        return build_special(models[get_num(model_name)-1])
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented')

def build_special(config):
    model_name, *args = config
    if model_name == 'mlp1':
        return MLPDetector(1, *args)
    if model_name == 'mlp2':
        return MLPDetectorv2(1, *args)
    if model_name == 'yolo5':
        return YoloNet5(1, *args)
    if model_name == 'yolo8':
        return YoloNet8(1, *args)
    if model_name == 'yolophi':
        return YoloNetPhi(1, *args)



def get_num(tx):
    try:
        return int(tx)+1
    except:
        return 0

models = [['yolo5',
  0.1,
  0.59,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['yolo5',
  0.25,
  0.1,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['mlp2',
  False,
  1.2,
  5,
  2,
  [[10, 13, 16, 30, 33, 23], [30, 61, 17, 31, 59, 119]]],
 ['yolophi',
  0.4,
  0.15000000000000002,
  0.2,
  7,
  0.94,
  True,
  False,
  [[10, 13, 16, 30, 33, 23],
   [30, 61, 17, 31, 59, 119],
   [116, 90, 156, 198, 60, 110]]],
 ['mlp2',
  False,
  2.2,
  8,
  2,
  [[1, 4, 4, 8, 16, 16, 2, 2], [44, 44, 10, 30, 50, 70, 60, 110]]],
 ['yolo8',
  0.2,
  0.1,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['mlp2',
  False,
  2.2,
  3,
  0,
  [[10, 13, 16, 30, 33, 23], [30, 61, 17, 31, 59, 119]]],
 ['yolo5',
  0.5,
  0.1,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['yolo8',
  0.3,
  0.38,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['yolo8',
  0.39999999999999997,
  0.38,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['yolo8',
  0.1,
  0.44999999999999996,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]],
 ['yolo8',
  0.9,
  0.1,
  [[2, 6, 8, 8, 5, 15], [5, 15, 40, 40, 20, 35], [100, 100, 40, 80, 80, 60]]]]