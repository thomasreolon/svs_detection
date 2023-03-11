import torch
import numpy as np
from models._head import box_iou, xywh2xyxy, Detect


def compute_map(gts, y_pred):
    """ computes MaP from a batch of images

    Arguments:
        gts     Array[N,6]          --> groundtruths from dataset
        labels  list( Array[Mi,5] ) --> first element returned by detection head when mode = eval()
    Returns:
        float MaP
    """

    gts = gts.cpu()
    if not isinstance(y_pred, torch.Tensor):
        y_pred = [torch.cat((torch.from_numpy(x),torch.ones(x.shape[0],1),torch.zeros(x.shape[0],1)),dim=1) for x in y_pred]
    else:
        y_pred = y_pred.cpu()
        y_pred = Detect.postprocess(y_pred, 0.4, 0.4)
    stats = []
    for i,pred in enumerate(y_pred):
        gt = gts[gts[:,0]==i,1:]
        pred = pred/torch.tensor([160.,128,160,128,1,1])
        pred[:,:4] = xywh2xyxy(pred[:,:4])
        gt[:,1:]   = xywh2xyxy(gt[:,1:])
        update_map(pred , gt, stats)
    map = get_map(stats) [-1]
    return map

def get_map(stats):
    """first collect the stats with UPDATE_MAP then use this function to compute the MaP"""
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    return ap_per_class(*stats) if len(stats) else [0]*4

def update_map(detections, labels, stats):
    """ from yolo
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    device = detections.device
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    if len(labels) and not len(detections):
        correct = torch.zeros(0, niou, dtype=torch.bool, device=device)  # init
        stats.append((correct, *torch.zeros((3, 0), device=device)))
    elif not len(labels):
        correct = torch.zeros(len(detections), niou, dtype=torch.bool, device=device)  # init
        stats.append((correct, detections[:, 4], detections[:, 5], labels[:, 0]))
    else:
        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
        pbox = xywh2xyxy(detections[:, :4])  # target boxes
        detections = torch.cat((pbox, detections[:,4:]), 1)
        correct = process_batch(detections, labelsn, iouv)
        stats.append((correct, detections[:, 4], detections[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

def process_batch(detections, labels, iouv):
    "yolo code"
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct






def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)


    if len(f1)>0:
        i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()  # true positives
        mp, mr, map50, map = p.mean(), r.mean(), ap[:, 0].mean(), ap.mean()
    else:
        mp, mr, map50, map = [1]*4
    
    return mp, mr, map50, map


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

