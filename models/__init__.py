from ._loss import ComputeLoss
from .mlp import MLPDetector
from .mlp_v2 import MLPDetectorv2
from .yolov5 import YoloNetSmall, YoloNetPhi

def build(model_name, ch_in = 1):
    if model_name == 'mlp1':
        return MLPDetector(ch_in)
    if model_name == 'mlp2':
        return MLPDetectorv2(ch_in)
    if model_name == 'yolos':
        return MLPDetectorv2(ch_in)
    if model_name == 'yolophi':
        return MLPDetectorv2(ch_in)
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented')

