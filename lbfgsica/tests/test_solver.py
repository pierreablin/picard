import numpy as np
from lbfgsica import lbfgs_ica


def test_lbgfs_solver():
    N, T = 15, 5000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    X = np.dot(A, S)
    Y, W = lbfgs_ica(X, verbose=True)


def test_lbgfs_h2_solver():
    N, T = 2, 1000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    X = np.dot(A, S)
    Y, W = lbfgs_ica(X, precon=2)


def test_lbgfs_crash():
    N, T = 2, 1000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    X = np.dot(A, S)
    try:
        Y, W = lbfgs_ica(X, precon=18)
    except:
        pass
