#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Example Diseases')

import numpy as np
from scipy.sparse import lil_matrix
from scipy.io.matlab import loadmat
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from copacar import copacar_sgd


def predict_copacar_sgd(T):
    A, R, _, _, = copacar_sgd(
        T, 10, conv=0.1, lambda_A=0,
        lambda_R=0, n_jobs=2,
        max_iter=20, sample_size=500,
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


def innerfold(T, mask_idx, target_idx, e, k, r):
    mask_idx = np.unravel_index(mask_idx, (e, e))
    target_idx = np.unravel_index(target_idx, (e, e))

    # set values to be predicted to zero
    Tc = [Ti.copy() for Ti in T]
    for i in range(len(mask_idx[0])):
        Tc[r][mask_idx[0][i], mask_idx[1][i]] = 0
        Tc[r][mask_idx[1][i], mask_idx[0][i]] = 0
    Tc1 = [np.array(Ti.todense()) for Ti in Tc]

    # predict unknown values
    P = predict_copacar_sgd(Tc1)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    idx0, idx1 = target_idx
    roc_auc = roc_auc_score(GROUND_TRUTH[idx0, idx1, r], P[idx0, idx1, r])
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[idx0, idx1, r], P[idx0, idx1, r])
    return roc_auc, auc(recall, prec)


if __name__ == '__main__':
    # seed data
    np.random.seed(0)

    # load data
    mat = loadmat('data/diseases.mat')
    K = np.array(mat['K'][:, :, :], np.float32)
    e, k = K.shape[0], K.shape[2]
    SZ = e * e

    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()
    # construct data array
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    _log.info('Data size: %d x %d x %d | No. of classes: %d' % (
        T[0].shape + (len(T),) + (k,)))

    for r in range(k):
        # Do cross-validation for each relation
        _log.info('Cross-validation for relation: %d/%d' % (r + 1, k))
        FOLDS = 10
        IDX = list(range(SZ))
        np.random.shuffle(IDX)

        fsz = int(SZ / FOLDS)
        offset = 0
        PR_AUC_test = np.zeros(FOLDS)
        ROC_AUC_test = np.zeros(FOLDS)
        for f in range(FOLDS):
            # account for the symmetry of relations
            idx_test = IDX[offset:offset + fsz]
            ROC_AUC_test[f], PR_AUC_test[f] = innerfold(T, idx_test, idx_test, e, k, r)
            _log.info('[Fold:%2d] test ROC-AUC: %0.5f | test PR-AUC: '
                      '%0.5f' % (f, ROC_AUC_test[f], PR_AUC_test[f]))
            offset += fsz

        _log.info('AUC-PR Test Mean / Std: %f / %f' % (PR_AUC_test.mean(), PR_AUC_test.std()))
        _log.info('AUC-ROC Test Mean / Std: %f / %f' % (ROC_AUC_test.mean(), ROC_AUC_test.std()))
