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

