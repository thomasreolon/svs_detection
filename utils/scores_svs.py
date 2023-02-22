import numpy as np
import cv2
import torch
from . import xywh2xyxy

def get_scores(svs, gt):
    boxes = (gt[:,2:] * torch.tensor([svs.shape[1], svs.shape[0], svs.shape[1], svs.shape[0]]))
    boxes = xywh2xyxy(boxes).int().tolist()

    # activate where target
    truth = np.zeros_like(svs).astype(bool)
    for x1,y1,x2,y2 in boxes:
        truth[y1:y2, x1:x2] = True
    # get scores
    scores = []
    for score in [score_precision, score_recall, score_stability, score_cc]:
        scores.append(score(svs.copy(), truth))
    scores.append(2*scores[0]*scores[1]/(1e-4+scores[0]+scores[1])) #F1
    return scores

def score_precision(svs, truth):
    # activate only where target
    score = svs[truth].sum() / (1e-4+svs.sum())
    return score

def score_recall(svs, truth):
    # activate where target
    score = svs[truth].sum()/255 / (1e-4+truth.sum())
    return score

_memory = [0]
def score_stability(svs, truth):
    # no flickering
    out = int(svs[~truth].sum())
    score = 2.**(-abs(out-_memory[0])/20480) /2
    _memory[0] = out
    return score

def score_cc(svs, truth):
    # don't cut people
    svs[~truth] = 0
    truth = truth.astype(np.uint8) * 255
    true_n_cc, _, _, _ = cv2.connectedComponentsWithStats(truth, 5, cv2.CV_32S)
    pred_n_cc, _, _, _ = cv2.connectedComponentsWithStats(svs, 5, cv2.CV_32S)
    score = 2.**(-abs(pred_n_cc-true_n_cc)/(true_n_cc)) /2
    return score


