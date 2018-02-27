# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_equal

from picard import picard, Density
from picard.densities import Tanh, Exp, Cube, check_density


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
    for j, fun in enumerate([Tanh(params=dict(alpha=0.5)), 'cube']):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(X.copy(), fun=fun, ortho=False, verbose=True)
        if fun == 'tanh':
            fun = Tanh()
        elif fun == 'exp':
            fun = Exp()
        elif fun == 'cube':
            fun = Cube()
        # Get the final gradient norm
        psiY = fun.score_and_der(Y)[0]
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        err_msg = 'fun %s, gradient norm greater than tol' % names[j]
        assert_allclose(G, np.zeros((N, N)), atol=1e-7,
                        err_msg=err_msg)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        assert_equal(K.shape, A.shape)
        WA = W.dot(K).dot(A)
        WA = get_perm(WA)[1]  # Permute and scale
        err_msg = 'fun %s, wrong unmixing matrix' % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=5e-2,
                        err_msg=err_msg)


def test_shift():
    N, T = 5, 10000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    offset = rng.randn(N)
    X = np.dot(A, S) + offset[:, None]
    _, W, Y, X_mean = picard(X.copy(), ortho=False, verbose=2, whiten=False,
                             return_X_mean=True)
    assert_allclose(offset, X_mean, rtol=0, atol=0.1)
    WA = W.dot(A)
    WA = get_perm(WA)[1]
    assert_allclose(WA, np.eye(N), rtol=0, atol=0.1)


def test_picardo():
    N, T = 3, 20000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    names = ['tanh', 'exp', 'cube']
    for fun in names:
        K, W, Y = picard(X.copy(), fun=fun, ortho=True, verbose=2)
        if fun == 'tanh':
            fun = Tanh()
        elif fun == 'exp':
            fun = Exp()
        elif fun == 'cube':
            fun = Cube()
        # Get the final gradient norm
        psiY = fun.score_and_der(Y)[0]
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        G = (G - G.T) / 2.  # take skew-symmetric part
        err_msg = 'fun %s, gradient norm greater than tol' % fun
        assert_allclose(G, np.zeros((N, N)), atol=1e-7,
                        err_msg=err_msg)
        assert_equal(Y.shape, X.shape)
        assert_equal(W.shape, A.shape)
        assert_equal(K.shape, A.shape)
        WA = W.dot(K).dot(A)
        WA = get_perm(WA)[1]  # Permute and scale
        err_msg = 'fun %s, wrong unmixing matrix' % fun
        assert_allclose(WA, np.eye(N), rtol=0, atol=5e-2,
                        err_msg=err_msg)


def test_dimension_reduction():
    N, T = 5, 10000
    n_components = 3
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    K, W, Y = picard(X.copy(), n_components=n_components, ortho=False)
    assert_equal(K.shape, (n_components, N))
    assert_equal(W.shape, (n_components, n_components))
    assert_equal(Y.shape, (n_components, T))
    K, W, Y = picard(X.copy(), n_components=n_components, ortho=False,
                     whiten=False)


def test_bad_custom_density():
    def log_lik(self, Y):
        return Y ** 4 / 4

    def score_and_der(self, Y):
        return Y ** 3, 3 * Y ** 2 + 2.

    fun = Density(log_lik, score_and_der=score_and_der)
    X = np.random.randn(2, 10)
    try:
        picard(X, fun=fun, check_fun=True)
    except AssertionError:
        pass
    else:
        raise(AssertionError, 'Bad function undetected')


def test_fun():
    for fun in [Tanh(), Exp(), Cube()]:
        check_density(fun)
