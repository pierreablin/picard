# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)
import numbers

import numpy as np

from scipy.linalg import expm


def gradient(Y, psiY):
    '''
    Compute the gradient for the current signals
    '''
    _, T = Y.shape
    return np.inner(psiY, Y) / float(T)


def proj_hessian_approx(Y, psidY_mean, G):
    '''
    Computes the projected Hessian approximation.
    '''
    N, _ = Y.shape
    diag = psidY_mean[:, None] * np.ones(N)[None, :]
    off_diag = np.diag(G)
    return 0.5 * (diag + diag.T - off_diag[:, None] - off_diag[None, :])


def regularize_hessian(h, l):
    '''
    Clips the eigenvalues of h to l
    '''
    h[h < l] = l
    return h


def solve_hessian(G, h):
    '''
    Returns the inverse Hessian times G
    '''
    return G / h


def loss(Y, signs, density):
    '''
    Returns the loss function
    '''
    output = 0.
    _, T = Y.shape
    for y, s in zip(Y, signs):
        output += s * np.mean(density.log_lik(y))
    return output


def l_bfgs_direction(G, h, s_list, y_list, r_list):
    q = G.copy()
    a_list = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        alpha = r * np.sum(s * q)
        a_list.append(alpha)
        q -= alpha * y
    z = solve_hessian(q, h)
    for s, y, r, alpha in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (alpha - beta) * s
    return -z


def line_search(Y, signs, density, direction, current_loss, ls_tries):
    '''
    Performs a backtracking line search, starting from Y and W, in the
    direction direction.
    '''
    alpha = 1.
    if current_loss is None:
        current_loss = loss(Y, signs, density)
    for _ in range(ls_tries):
        Y_new = np.dot(expm(alpha * direction), Y)
        new_loss = loss(Y_new, signs, density)
        if new_loss < current_loss:
            return True, Y_new, new_loss, alpha
        alpha /= 2.
    else:
        return False, Y_new, new_loss, alpha


def permute(A):
    '''Get a permutation to diagonalize and scale a matrix

    Parameters
    ----------
    A : ndarray, shape (n_features, n_features)
        A matrix close from a permutation and scale matrix.

    Returns
    -------
    A : ndarray, shape (n_features, n_features)
        A permuted matrix.
    '''
    A = A.copy()
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
    return A


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = np.linalg.eigh(np.dot(W, W.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_par(X, fun, max_iter, w_init, verbose):
    """Parallel FastICA.
    Used internally by FastICA --main loop
    """
    if verbose:
        print('Running %d iterations of FastICA...' % max_iter)
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = fun.score_and_der(np.dot(W, X))
        g_wtx = g_wtx.mean(axis=1)
        W = _sym_decorrelation(np.dot(gwtx, X.T) / p_ -
                               g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
    if verbose:
        print('Running Picard...')
    return W
