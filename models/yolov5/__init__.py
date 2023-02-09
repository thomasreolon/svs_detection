from .models.yolo import YoloNet
import pathlib


def YoloNet5(ch_in):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/yolov5n.yaml'
    return YoloNet(cfg, ch_in, 1)

def YoloNet8(ch_in):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/yolov8n.yaml'
    return YoloNet(cfg, ch_in, 1)


def YoloNetPhi(ch_in):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/phi10.yaml'
    return YoloNet(cfg, ch_in, 1)


