from operator import le
from sklearn import metrics
from collections import defaultdict
import json
import numpy as np


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.7).astype(np.int)
    mat=metrics.matthews_corrcoef(target, pred)
    #acc = metrics.balanced_accuracy_score(target, pred)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)
    fpr,tpr,thr=metrics.roc_curve(target, pred)
    print(tpr)
    print(1-fpr)
    return acc, auroc, f1_score, precision, recall, int_ap, ap,pred,target

'''def do_compute_metrics(probas_pred, target):
    #pred = (probas_pred >= 0.5).astype(np.int)
    pred = np.array([np.where(j==np.max(j)) for j in probas_pred]).reshape(-1)
    #print(pred)
    mat=metrics.matthews_corrcoef(target, pred)
    #acc = metrics.balanced_accuracy_score(target, pred)
    acc = metrics.accuracy_score(target, pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    fpr,tpr,thr=metrics.roc_curve(target, pred)
    print(tpr)
    print(1-fpr)
    return acc, f1_score, precision, recall,pred,target'''


