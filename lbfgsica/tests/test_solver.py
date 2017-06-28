import numpy as np
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_allclose

from lbfgsica import lbfgs_ica


def test_lbgfs_solver():
    N, T = 2, 10000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)

    for precon in [1, 2]:
        Y, W = lbfgs_ica(X, precon=2, verbose=True)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        WA = np.abs(np.dot(W, A))
        WA /= np.max(WA, axis=1, keepdims=True)
        assert_allclose(WA, np.eye(N), rtol=1e-2, atol=1e-2)


def test_lbgfs_crash():
    N, T = 2, 1000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    assert_raises(ValueError, lbfgs_ica, X, precon=18)
