# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:41:51 2019

@author: admin
""" 
from sklearn import metrics
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cluster_acc(y_true, y_pred): 
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true,y_pred):
    acc = np.round(cluster_acc(y_true, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    return acc, nmi, ari


if __name__ == '__main__':
    y_true = np.array([1,1,2,2,3,3,3,3])
    y_pred = np.array([2,2,1,1,1,1,1,2])
    acc, nmi, ari = evaluate(y_true, y_pred)
    print('acc:%.5f, nmi:%.5f, ari:%.5f'%(acc, nmi, ari))
 