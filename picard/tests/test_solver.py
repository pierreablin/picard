import numpy as np
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_allclose

from picard import picard


def get_perm(A):
    '''
    returns a list of permutaion indices ind such that A[idx, idx] is close to
    identity, as well as the output matrix
    '''
    n = A.shape[0]
    idx = np.arange(n)
    done = False
    while not done:
        done = True
        for i in range(n):
            for j in range(i):
                if A[i, i] ** 2 + A[j, j] ** 2 < A[i, j] ** 2 + A[j, i] ** 2:
                    A[(i, j), :] = A[(j, i), :]
                    idx[i], idx[j] = idx[j], idx[i]
                    done = False
    A /= np.diag(A)
    order_sort = np.argsort(np.sum(np.abs(A), axis=0))
    A = A[order_sort, :]
    A = A[:, order_sort]
    return idx, A


def test_picard():
    N, T = 2, 10000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    for precon in [1, 2]:
        Y, W = picard(X, precon=precon, verbose=True)
        # Get the final gradient norm
        G = np.inner(np.tanh(Y / 2.), Y) / float(T) - np.eye(N)
        assert_allclose(G, np.zeros((N, N)), atol=1e-7)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        WA = np.dot(W, A)
        WA = get_perm(WA)[1]  # Permute and scale
        assert_allclose(WA, np.eye(N), rtol=1e-2, atol=1e-2)


def test_lbgfs_crash():
    N, T = 2, 1000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    assert_raises(ValueError, picard, X, precon=18)
