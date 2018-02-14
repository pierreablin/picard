# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose

from nose.tools import assert_equal

from picard import picard, tanh, exp, cube


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
    N, T = 3, 20000
    rng = np.random.RandomState(42)
    names = ['tanh', 'cube']
    for j, density in enumerate([tanh(params=dict(alpha=0.5)), 'cube']):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(X.copy(), density=density, ortho=False, verbose=True)
        if density == 'tanh':
            density = tanh()
        elif density == 'exp':
            density = exp()
        elif density == 'cube':
            density = cube()
        # Get the final gradient norm
        G = np.inner(density.score(Y), Y) / float(T) - np.eye(N)
        err_msg = 'density %s, gradient norm greater than tol' % names[j]
        assert_allclose(G, np.zeros((N, N)), atol=1e-7,
                        err_msg=err_msg)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        assert_equal(K.shape, A.shape)
        WA = W.dot(K).dot(A)
        WA = get_perm(WA)[1]  # Permute and scale
        err_msg = 'density %s, wrong unmixing matrix' % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=5e-2,
                        err_msg=err_msg)


def test_picardo():
    N, T = 3, 20000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    names = ['tanh', 'exp', 'cube']
    for density in names:
        K, W, Y = picard(X.copy(), density=density, ortho=True, verbose=2)
        if density == 'tanh':
            density = tanh()
        elif density == 'exp':
            density = exp()
        elif density == 'cube':
            density = cube()
        # Get the final gradient norm
        G = np.inner(density.score(Y), Y) / float(T) - np.eye(N)
        G = (G - G.T) / 2.  # take skew-symmetric part
        err_msg = 'density %s, gradient norm greater than tol' % density
        assert_allclose(G, np.zeros((N, N)), atol=1e-7,
                        err_msg=err_msg)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        assert_equal(K.shape, A.shape)
        WA = W.dot(K).dot(A)
        WA = get_perm(WA)[1]  # Permute and scale
        err_msg = 'density %s, wrong unmixing matrix' % density
        assert_allclose(WA, np.eye(N), rtol=0, atol=5e-2,
                        err_msg=err_msg)


def test_density():
    for density in [tanh(), exp(), cube()]:
        density.check()
