#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Kinships')

import numpy as np
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from copacar import copacar_sgd


def predict_copacar_sgd(T):
    A, R, _, _, = copacar_sgd(
        T, 10, conv=0.1, lambda_A=1e-1,
        lambda_R=1e-1, n_jobs=50,
        max_iter=100,
    )
    n = A.shape[0]
    P = np.zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = np.dot(A, np.dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = np.linalg.norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P


def innerfold(T, mask_idx, target_idx, e, k, sz):
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    Tc = [Ti.copy() for Ti in T]
    for i in range(len(mask_idx[0])):
        Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0
    Tc1 = [np.array(Ti.todense()) for Ti in Tc]

    # predict unknown values
    P = predict_copacar_sgd(Tc1)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    roc_auc = roc_auc_score(GROUND_TRUTH[target_idx], P[target_idx])
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    return roc_auc, auc(recall, prec)


if __name__ == '__main__':
    # seed data
    np.random.seed(0)

    # load data
    mat = loadmat('data/alyawarradata.mat')
    K = np.array(mat['Rs'][:, :, :], np.float32)

    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k

    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()
    # construct data array
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    _log.info('Data size: %d x %d x %d | No. of classes: %d' % (
        T[0].shape + (len(T),) + (k,)))

    # Do cross-validation
    FOLDS = 10
    IDX = list(range(SZ))
    np.random.shuffle(IDX)

    fsz = int(SZ / FOLDS)
    offset = 0
    PR_AUC_test = np.zeros(FOLDS)
    ROC_AUC_test = np.zeros(FOLDS)
    for f in range(FOLDS):
        idx_test = IDX[offset:offset + fsz]
        ROC_AUC_test[f], PR_AUC_test[f] = innerfold(T, idx_test, idx_test, e, k, SZ)
        _log.info('[Fold:%2d] test ROC-AUC: %0.5f | test PR-AUC: '
                  '%0.5f' % (f, ROC_AUC_test[f], PR_AUC_test[f]))
        offset += fsz

    _log.info('AUC-PR Test Mean / Std: %f / %f' % (PR_AUC_test.mean(), PR_AUC_test.std()))
    _log.info('AUC-ROC Test Mean / Std: %f / %f' % (ROC_AUC_test.mean(), ROC_AUC_test.std()))
