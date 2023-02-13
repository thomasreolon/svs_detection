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
    if model_name == 'opt_mlp2':
        return build_special(['mlp2', False, 2.2, 6, 3, [[1, 4, 4, 8, 16, 16, 2, 2], [44, 44, 10, 30, 50, 70, 60, 110]]])
    if model_name == 'opt_yolo8':
        return build_special(['yolo8', 0.33, 0.5])
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

