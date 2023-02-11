from .models.yolo import YoloNet
import pathlib
import yaml


def load(cfg, depth_multiple=None, width_multiple=None):
    with open(cfg, 'r') as ff:
        cfg = yaml.safe_load(ff)
    if depth_multiple: cfg['depth_multiple'] = depth_multiple
    if width_multiple: cfg['width_multiple'] = width_multiple
    return cfg


def YoloNet5(ch_in, dm=None, wm=None):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/yolov5n.yaml'
    cfg = load(cfg, dm, wm)
    return YoloNet(cfg, ch_in, 1)

def YoloNet8(ch_in, dm=None, wm=None):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/yolov8n.yaml'
    cfg = load(cfg, dm, wm)
    return YoloNet(cfg, ch_in, 1)

def YoloNetPhi(ch_in, dm=None, wm=None, a=0.35, b0=7, b=1, se=False, c2=True):
    cfg = str(pathlib.Path(__file__).parent.resolve()) + '/models/phi10.yaml'
    cfg = load(cfg, dm, wm)
    phi = cfg['backbone'][0][3]
    phi[2] = a
    phi[3] = b0
    phi[4] = b
    phi[5] = se
    phi[6] = c2
    return YoloNet(cfg, ch_in, 1)


