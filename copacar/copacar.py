# coding: utf-8
# Copyright (C) 2015 Marinka Zitnik <marinka.zitnik@fri.uni-lj.si>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import time

import numpy as np
from sklearn import metrics
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

__version__ = "0.1"
__all__ = ['copacar']

_DEF_MAXITER = 20
_DEF_CONV = 2e-1
_DEF_LMBDA = 0
_DEF_SAMPLEEPOCH = 1
_DEF_ARMIJO = 1e-3
_DEF_INITGAMMA = 1e-1
_DEF_SAMPLESIZE = 100
_DEF_NJOBS = 1

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('COPACAR')


def copacar(X, rank, **kwargs):
    """
    Parameters
    ----------
    X : list
        List of frontal slices X_k of the tensor X.
        The shape of each X_k is ('N', 'N').
    rank : int
        Rank of the factorization
    lmbdaA : float, optional
        Regularization parameter for A factor matrix. 0 by default.
    lmbdaR : float, optional
        Regularization parameter for R_k factor matrices. 0 by default.
    lmbdaV : float, optional
        Regularization parameter for V_l factor matrices. 0 by default.
    max_iter : int, optional
        Maximium number of iterations of the algorithm. 500 by default.
    sample_epoch : int, optional
        Used with stochastic optimization of the gradient. The number of
        iterations that model is optimized against a particular data subsample.
        5 by default.
    conv : float, optional
        Stop when residual of factorization is less than conv. 1e-5 by default.
    sample_epoch : int
        Number of consecutive algorithm iterations run using the same data sample.
        1 by default.
    armijo : float
        Gap in the Armijo-Goldstein step size control. 1e-3 by default.
    init_gamma : float
        Maximum candidate step in the Armijo-Goldstein step. 1e-1 by default.
    sample_size : init
        Data subsample size, the actual size is 2*sample_size. 100 by default.
    n_jobs : int
        Number of jobs to run in parallel. 1 by default.

    Returns
    -------
    A : ndarray
        array of shape ('N', 'rank') corresponding to the factor matrix A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the
        factor matrices R_k
    itr : int
        number of iterations until convergence
    exectimes : ndarray
        execution times to compute the updates in each iteration
    """

    # ------------ init options ----------------------------------------------
    np.random.seed(0)
    max_iter = kwargs.pop('max_iter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    sample_epoch = kwargs.pop('sample_epoch', _DEF_SAMPLEEPOCH)
    armijo = kwargs.pop('armijo', _DEF_ARMIJO)
    init_gamma = kwargs.pop('gamma', _DEF_INITGAMMA)
    sample_size = kwargs.pop('sample_size', _DEF_SAMPLESIZE)
    dtype = kwargs.pop('dtype', np.float)
    n_jobs = kwargs.pop('n_jobs', _DEF_NJOBS)

    # ------------- check input ----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have the same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')

    n = sz[0]
    k = len(X)

    _log.debug(
        '[Config] rank: %d | max_iter: %d | conv: %7.1e | lmbda: %7.1e' %
        (rank, max_iter, conv, lmbdaA)
    )
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X to CSR -----------------------------------------------
    for i in range(k):
        if issparse(X[i]):
            X[i] = X[i].tocsr()
            X[i].sort_indices()

    # ---------- initialize A ------------------------------------------------
    _log.debug('Initializing A')
    S = csr_matrix((n, n), dtype=dtype)
    for i in range(k):
        S = S + X[i]
        S = S + X[i].T
    _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
    A = np.array(A, dtype=dtype)
    A /= np.linalg.norm(A)

    # ------- initialize R ---------------------------------------------------
    R = []
    invA = np.linalg.pinv(A)
    for i in range(len(X)):
        Ri = np.dot(invA, np.dot(X[i], invA.T))
        R.append(Ri / np.linalg.norm(Ri))

    # ------ initialize indices ----------------------------------------------
    e = X[0].shape[0]
    SZ = e * e
    fsz = min(sample_size, SZ / sample_size)

    indices1 = zip(*[np.where(X[i] == 0) for i in range(len(X))])
    indices1 = [np.concatenate(ind) for ind in indices1]
    IDX1 = list(range(len(indices1[0])))
    indices2 = zip(*[np.nonzero(X[i]) for i in range(len(X))])
    indices2 = [np.concatenate(ind) for ind in indices2]
    IDX2 = list(range(len(indices2[0])))

    idx_group1 = None
    idx_group2 = None

    # ------ compute factorization -------------------------------------------
    exectimes = []
    fit_values = []
    itr = 0
    for itr in range(max_iter):
        tic = time.time()

        if itr % sample_epoch == 0:
            np.random.shuffle(IDX1)
            np.random.shuffle(IDX2)
            idx_group1 = indices1[0][IDX1[:fsz]], indices1[1][IDX1[:fsz]]
            idx_group2 = indices2[0][IDX2[:fsz]], indices2[1][IDX2[:fsz]]

        A = _updateA(
            X, A, R, idx_group1, idx_group2, lmbdaA, lmbdaR, init_gamma, armijo, n_jobs)
        R = _updateR(
            X, A, R, idx_group1, idx_group2, lmbdaR, lmbdaA, init_gamma, armijo, n_jobs)

        toc = time.time()
        exectimes.append(toc - tic)

        auc = []
        for i in range(len(X)):
            for group in [idx_group1, idx_group2]:
                xidx, yidx = group
                if len(np.unique(X[i][xidx, yidx])) == 1:
                    continue
                P = np.dot(A, np.dot(R[i], A.T))
                auc.append(metrics.roc_auc_score(X[i][xidx, yidx], P[xidx, yidx]))
        fit = float(np.mean(auc))
        fit_values.append(fit)

        _log.info('[%3d] fit: %0.5f | secs: %.5f' % (itr, fit, exectimes[-1]))
        diff = np.diff(fit_values)[-3:]
        stop = float(np.mean(np.abs(diff)))
        if np.alltrue(diff < 0) and stop > conv:
            _log.info('[%3.5f > %3.5f] Early termination' % (stop, conv))
            break

    return A, R, itr + 1, np.array(exectimes)


# ------------------ Update A ------------------------------------------------
def _updateA(X, A, R, IDX1, IDX2, lmbdaA, lmbdaR, init_gamma, armijo, n_jobs):
    """Update step for A"""
    _log.debug('Updating A lambda A: %s' % str(lmbdaA))
    Grad = np.zeros_like(A)

    uniq_IDX21 = np.unique(IDX2[1])
    indices_IDX21 = [np.where(IDX2[1] == el)[0] for el in uniq_IDX21]

    uniq_IDX20 = np.unique(IDX2[0])
    indices_IDX20 = [np.where(IDX2[0] == el)[0] for el in uniq_IDX20]

    parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e5, backend="multiprocessing")
    task_iter = (delayed(__updateA)(
        X, A, R, IDX1, IDX2, uniq_IDX20, uniq_IDX21, indices_IDX20, indices_IDX21, i)
                 for i in range(len(X)))
    grad_count = parallelizer(task_iter)

    for i in range(len(X)):
        Grad += grad_count[i][0] / grad_count[i][1][:, None]
    Grad = Grad + lmbdaA * A

    gamma = init_gamma
    Atmp = A - gamma * Grad
    norm_grad = np.linalg.norm(Grad, 2)**2
    gap = armijo * gamma * norm_grad
    while True:
        ls = _pairwise_rank_bound(X, Atmp, R, IDX1, IDX2, lmbdaA, lmbdaR)
        rs = _pairwise_rank_bound(X, A, R, IDX1, IDX2, lmbdaA, lmbdaR) - gap
        if ls > rs:
            gamma /= 2.
            Atmp = A - gamma * Grad
            gap = armijo * gamma * norm_grad
        else:
            break

    step = gamma
    A -= step * (Grad + lmbdaA * A)
    return A


def __updateA(X, A, R, IDX1, IDX2, uniq_IDX20, uniq_IDX21,
              indices_IDX20, indices_IDX21, i):
    Grad = np.zeros_like(A)
    n_count = np.ones(A.shape[0])

    RkA = np.dot(R[i], A.T)
    RtkA = np.dot(R[i].T, A.T)
    X_hat = np.dot(A, np.dot(R[i], A.T))

    for a in range(len(IDX1[0])):
        weights, weights2 = _grad_A(X_hat, X[i], IDX1[0][a], IDX1[1][a], IDX2)

        weight_sum = np.sum(weights)
        weight_count = np.sum(weights != 0)
        Grad[IDX1[0][a]] += weight_sum * RkA[:, IDX1[1][a]]
        Grad[IDX1[1][a]] += weight_sum * RtkA[:, IDX1[0][a]]

        n_count[IDX1[0][a]] += weight_count
        n_count[IDX1[1][a]] += weight_count

        TT1 = np.multiply(weights2, RkA[:, IDX2[1]])
        TT2 = np.multiply(weights2, RtkA[:, IDX2[0]])

        for u in range(len(uniq_IDX21)):
            Grad[uniq_IDX21[u], :] += np.sum(TT2[:, indices_IDX21[u]], axis=1)
            n_count[uniq_IDX21[u]] += weight_count
        for u in range(len(uniq_IDX20)):
            Grad[uniq_IDX20[u], :] += np.sum(TT1[:, indices_IDX20[u]], axis=1)
            n_count[uniq_IDX20[u]] += weight_count
    return Grad, n_count


# ---------------- Gradient for A --------------------------------------------
def _grad_A(P, T, a1, a2, IDX2):
    rel_true = T[a1, a2] - T[IDX2]
    exp_p = np.exp(np.minimum(P[a1, a2], 10))
    exp_idx2 = np.exp(np.minimum(P[IDX2], 10))
    denom = exp_p + exp_idx2 + 1e-5
    der1 = - np.divide(np.multiply(rel_true, exp_idx2), denom)
    der2 = np.divide(rel_true * exp_p, denom)
    return der1, der2


# ------------------ Update R ------------------------------------------------
def _updateR(X, A, R, IDX1, IDX2, lmbdaR, lmbdaA, init_gamma, armijo, n_jobs):
    _log.debug('Updating R lambda R: %s' % str(lmbdaR))

    parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e5, backend="multiprocessing")
    task_iter = (delayed(__updateR)(X, A, R, IDX1, IDX2, i) for i in range(len(X)))
    grads_count = parallelizer(task_iter)

    Grads = [grads_count[i][0] / grads_count[i][1] + lmbdaR * R[i] for i in range(len(X))]

    for i in range(len(X)):
        gamma = init_gamma
        Ritmp = R[i] - gamma * Grads[i]
        norm_grad = np.linalg.norm(Grads[i], 2)**2
        gap = armijo * gamma * norm_grad
        while True:
            Rtmp = [Ri if j != i else Ritmp for j, Ri in enumerate(R)]
            ls = _pairwise_rank_bound(X, A, Rtmp, IDX1, IDX2, lmbdaA, lmbdaR, dom=[i])
            rs = _pairwise_rank_bound(X, A, R, IDX1, IDX2, lmbdaA, lmbdaR, dom=[i]) - gap
            if ls > rs:
                gamma /= 2.
                Ritmp = R[i] - gamma * Grads[i]
                gap = armijo * gamma * norm_grad
            else:
                break

        step = gamma
        R[i] -= step * Grads[i]
    return R


def __updateR(X, A, R, IDX1, IDX2, i):
    X_hat = np.dot(A, np.dot(R[i], A.T))
    grad1 = np.zeros_like(R[0])
    grad = np.zeros_like(R[0])

    count = 1.
    for a in range(len(IDX1[0])):
        np.outer(A[IDX1[0][a], :], A[IDX1[1][a], :], grad1)
        weights = _grad_R(X_hat, X[i], IDX1[0][a], IDX1[1][a], IDX2)
        count += np.sum(weights != 0)
        grad_part1 = np.einsum('i,ij,ik->jk', weights, A[IDX2[0]], A[IDX2[1]])
        grad_part2 = np.sum(weights) * grad1
        grad += grad_part1 - grad_part2
    return grad, count


# ----------------- Gradient for R_k ----------------------------------------
def _grad_R(P, T, a1, a2, IDX2):
    rel_true = T[a1, a2] - T[IDX2]
    exp_idx2 = np.exp(np.minimum(P[IDX2], 10))
    exp_p = np.exp(np.minimum(P[a1, a2], 10))
    der = exp_idx2 / (exp_p + exp_idx2 + 1e-5)
    der1 = rel_true * der
    return der1


# --------------- Objective function -----------------------------------------
def _pairwise_rank(X, A, R, IDX):
    auc = []
    for i in range(len(X)):
        xidx, yidx = IDX
        P = np.dot(A, np.dot(R[i], A.T))
        auc.append(- metrics.roc_auc_score(X[i][xidx, yidx], P[xidx, yidx]))
    return np.sum(auc)


def _pairwise_rank_bound(X, A, R, IDX1, IDX2, lmbdaA, lmbdaR, dom=None):
    loss = 0
    dom = list(range(len(X))) if dom is None else dom
    for i in dom:
        P = np.dot(A, np.dot(R[i], A.T))
        rel_true = X[i][IDX1] - X[i][IDX2]
        rel_pred = P[IDX1] - P[IDX2]
        exp_p = np.exp(np.minimum(-rel_pred, 10))
        loss += np.sum(- np.multiply(rel_true, np.log(1. / (1. + exp_p))))
        loss += lmbdaR * np.linalg.norm(R[i])**2 / 2.
    loss += lmbdaA * np.linalg.norm(A)**2 / 2.
    return loss


if __name__ == '__main__':
    T = [(np.random.rand(50, 50) > 0.7).astype('int'),
         (np.random.rand(50, 50) < 0.3).astype('int')]
    A, R, _, _ = copacar(
        T, 5, conv=1e-3,
        lambda_A=10, lambda_R=10)
