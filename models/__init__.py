from ._loss import ComputeLoss
from .mini_phi import MiniYoloPHI
from .mini_phi2 import MiniYoloPHI2
from .mini_phi3 import MiniYoloPHI3
from .mlp_v2 import MLPDetectorv2
from .yolov5 import YoloNet5, YoloNet8, YoloNetPhi
from .blob_detector import BlobDetector

def build(model_name, ch_in = 1):
    if model_name == 'blob':
        return BlobDetector()
    if model_name == 'mlp2':
        return MLPDetectorv2(ch_in)
    if model_name == 'yolo5':
        return YoloNet5(ch_in)
    if model_name == 'yolo8':
        return YoloNet8(ch_in)
    if model_name == 'phiyolo': # Baseline
        return YoloNetPhi(ch_in) 
    if model_name == 'phisimple1':
        return MiniYoloPHI(ch_in)
    if model_name == 'phisimple2':
        return MiniYoloPHI2(ch_in)
    if model_name == 'phisimple3':
        return MiniYoloPHI3(ch_in)
    if model_name == 'phiyolo7K':
        return build_special(ch_in, ['yolophi', 0.15, 0.1, 0.05, 8, 0.82, False, True])
    if model_name == 'phiyolo77K':
        return build_special(ch_in, ['yolophi',  0.2,  0.55,  0.05,  9,  1.12,  False,  True])
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented')

def build_special(ch_in, config):
    model_name, *args = config
    if model_name == 'mlp2':
        return MLPDetectorv2(ch_in, *args)
    if model_name == 'yolo5':
        return YoloNet5(ch_in, *args)
    if model_name == 'yolo8':
        return YoloNet8(ch_in, *args)
    if model_name == 'yolophi':
        return YoloNetPhi(ch_in, *args)
