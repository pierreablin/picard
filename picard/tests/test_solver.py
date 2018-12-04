# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
import warnings
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_equal

from picard import picard, permute
from picard.densities import Tanh, Exp, Cube, check_density


def test_dimension_reduction():
    N, T = 5, 1000
    n_components = 3
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    K, W, Y = picard(X, n_components=n_components, ortho=False,
                     random_state=rng, max_iter=2)
    assert_equal(K.shape, (n_components, N))
    assert_equal(W.shape, (n_components, n_components))
    assert_equal(Y.shape, (n_components, T))
    with warnings.catch_warnings(record=True) as w:
        K, W, Y = picard(X, n_components=n_components, ortho=False,
                         whiten=False, max_iter=1)
        assert len(w) == 2


def test_dots():
    N, T = 5, 100
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    n_components = [N, 3]
    tf = [False, True]
    w_inits = [None, 'id']
    for n_component, ortho, whiten, w_init in product(n_components, tf, tf,
                                                      w_inits):
        if w_init == 'id':
            if whiten:
                w_init = np.eye(n_component)
            else:
                w_init = np.eye(N)
        with warnings.catch_warnings(record=True):
            K, W, Y, X_mean = picard(X, ortho=ortho, whiten=whiten,
                                     return_X_mean=True, w_init=w_init,
                                     n_components=n_component,
                                     random_state=rng, max_iter=2,
                                     verbose=False)
        if not whiten:
            K = np.eye(N)
        if ortho and whiten:
            assert_allclose(Y.dot(Y.T) / T, np.eye(n_component), atol=1e-8)
        Y_prime = np.dot(W, K).dot(X - X_mean[:, None])
        assert_allclose(Y, Y_prime, atol=1e-7)


def test_pre_fastica():
    N, T = 3, 1000
    rng = np.random.RandomState(42)
    names = ['tanh', 'cube']
    for j, fun in enumerate([Tanh(params=dict(alpha=0.5)), 'cube']):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(X, fun=fun, ortho=False, random_state=0,
                         fastica_it=10)
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
        WA = permute(WA)  # Permute and scale
        err_msg = 'fun %s, wrong unmixing matrix' % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1,
                        err_msg=err_msg)


def test_picard():
    N, T = 3, 1000
    rng = np.random.RandomState(42)
    names = ['tanh', 'cube']
    for j, fun in enumerate([Tanh(params=dict(alpha=0.5)), 'cube']):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(X, fun=fun, ortho=False, random_state=0)
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
        WA = permute(WA)  # Permute and scale
        err_msg = 'fun %s, wrong unmixing matrix' % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1,
                        err_msg=err_msg)


def test_extended():
    N, T = 4, 2000
    n = N // 2
    rng = np.random.RandomState(42)

    S = np.concatenate((rng.laplace(size=(n, T)),
                        rng.uniform(low=-1, high=1, size=(n, T))),
                       axis=0)
    print(S.shape)
    A = rng.randn(N, N)
    X = np.dot(A, S)
    K, W, Y = picard(X, ortho=False, random_state=0,
                     extended=True)
    assert_equal(Y.shape, X.shape)
    assert_equal(W.shape, A.shape)
    assert_equal(K.shape, A.shape)
    WA = W.dot(K).dot(A)
    WA = permute(WA)  # Permute and scale
    err_msg = 'wrong unmixing matrix'
    assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1,
                    err_msg=err_msg)


def test_shift():
    N, T = 5, 1000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    offset = rng.randn(N)
    X = np.dot(A, S) + offset[:, None]
    _, W, Y, X_mean = picard(X, ortho=False, whiten=False,
                             return_X_mean=True, random_state=rng)
    assert_allclose(offset, X_mean, rtol=0, atol=0.2)
    WA = W.dot(A)
    WA = permute(WA)
    assert_allclose(WA, np.eye(N), rtol=0, atol=0.2)
    _, W, Y, X_mean = picard(X, ortho=False, whiten=False,
                             centering=False,  return_X_mean=True,
                             random_state=rng)
    assert_allclose(X_mean, 0)


def test_picardo():
    N, T = 3, 2000
    rng = np.random.RandomState(4)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    names = ['tanh', 'exp', 'cube']
    for fastica_it in [None, 2]:
        for fun in names:
            print(fun)
            K, W, Y = picard(X, fun=fun, ortho=True, random_state=rng,
                             fastica_it=fastica_it, verbose=True,
                             extended=True)
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
            WA = permute(WA)  # Permute and scale
            err_msg = 'fun %s, wrong unmixing matrix' % fun
            assert_allclose(WA, np.eye(N), rtol=0, atol=0.1,
                            err_msg=err_msg)


def test_bad_custom_density():

    class CustomDensity(object):
        def log_lik(self, Y):
            return Y ** 4 / 4

        def score_and_der(self, Y):
            return Y ** 3, 3 * Y ** 2 + 2.

    fun = CustomDensity()
    X = np.random.randn(2, 10)
    try:
        picard(X, fun=fun, random_state=0)
    except AssertionError:
        pass
    else:
        raise(AssertionError, 'Bad function undetected')


def test_fun():
    for fun in [Tanh(), Exp(), Cube()]:
        check_density(fun)


def test_no_regression():
    n_tests = 10
    baseline = {}
    baseline['lap', True] = 17.
    baseline['lap', False] = 23.
    baseline['gauss', True] = 58.
    baseline['gauss', False] = 60.
    N, T = 10, 1000
    for mode in ['lap', 'gauss']:
        for ortho in [True, False]:
            n_iters = []
            for i in range(n_tests):
                rng = np.random.RandomState(i)
                if mode == 'lap':
                    S = rng.laplace(size=(N, T))
                else:
                    S = rng.randn(N, T)
                A = rng.randn(N, N)
                X = np.dot(A, S)
                _, _, _, n_iter = picard(X, return_n_iter=True,
                                         ortho=ortho, random_state=rng)
                n_iters.append(n_iter)
            n_mean = np.mean(n_iters)
            nb_mean = baseline[mode, ortho]
            err_msg = 'mode=%s, ortho=%s. %d iterations, expecting <%d.'
            assert n_mean < nb_mean, err_msg % (mode, ortho, n_mean, nb_mean)
