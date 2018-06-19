# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)
from copy import copy
import numpy as np

from .densities import Tanh


def picard_standard(X, density=Tanh(), m=7, maxiter=1000, tol=1e-7,
                    lambda_min=0.01, ls_tries=10, verbose=False):
    '''Runs the Picard algorithm

    The algorithm is detailed in::

        Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
        Faster independent component analysis by preconditioning with Hessian
        approximations
        ArXiv Preprint, June 2017
        https://arxiv.org/abs/1706.08171

    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the signals that have to be unmixed. N is the
        number of signals, T is the number of samples. X has to be centered

    m : int
        Size of L-BFGS's memory. Typical values for m are in the range 3-15

    maxiter : int
        Maximal number of iterations for the algorithm

    precon : 1 or 2
        Chooses which Hessian approximation is used as preconditioner.
        1 -> H1
        2 -> H2
        H2 is more costly to compute but can greatly accelerate convergence
        (See the paper for details).

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol.

    lambda_min : float
        Constant used to regularize the Hessian approximations. The
        eigenvalues of the approximation that are below lambda_min are
        shifted to lambda_min.

    ls_tries : int
        Number of tries allowed for the backtracking line-search. When that
        number is exceeded, the direction is thrown away and the gradient
        is used instead.

    verbose : boolean
        If true, prints the informations about the algorithm.

    Returns
    -------
    Y : array, shape (N, T)
        The estimated source matrix

    W : array, shape (N, N)
        The estimated unmixing matrix, such that Y = WX.
    '''
    # Init
    N, T = X.shape
    W = np.eye(N)
    Y = X
    s_list = []
    y_list = []
    r_list = []
    current_loss = _loss(Y, W, density)
    requested_tolerance = False
    G_norm = 1.
    for n in range(maxiter):
        # Compute the score function
        psiY, psidY = density.score_and_der(Y)
        # Compute the relative gradient
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        del psiY
        # Stopping criterion
        G_norm = np.max(np.abs(G))
        if G_norm < tol:
            requested_tolerance = True
            break
        # Update the memory
        if n > 0:
            s_list.append(direction)  # noqa
            y = G - G_old  # noqa
            y_list.append(y)
            r_list.append(1. / (np.sum(direction * y)))  # noqa
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                r_list.pop(0)
        G_old = G  # noqa
        # Find the L-BFGS direction
        direction = _l_bfgs_direction(Y, psidY, G, s_list, y_list, r_list,
                                      lambda_min)
        del psidY
        # Do a line_search in that direction:
        converged, new_Y, new_W, new_loss, direction =\
            _line_search(Y, W, density, direction, current_loss, ls_tries,
                         verbose)
        if not converged:
            direction = -G
            s_list, y_list, r_list = [], [], []
            _, new_Y, new_W, new_loss, direction =\
                _line_search(Y, W, density, direction, current_loss, 10, False)
        Y = new_Y
        W = new_W
        current_loss = new_loss
        if verbose:
            print('iteration %d, gradient norm = %.4g' %
                  (n + 1, G_norm))
    infos = dict(converged=requested_tolerance, gradient_norm=G_norm,
                 n_iterations=n)
    return Y, W, infos


def _loss(Y, W, density):
    '''
    Computes the loss function for Y, W
    '''
    loss = - np.linalg.slogdet(W)[1]
    for y in Y:
        loss += np.mean(density.log_lik(y))
    return loss


def _line_search(Y, W, density, direction, current_loss, ls_tries, verbose):
    '''
    Performs a backtracking line search, starting from Y and W, in the
    direction direction. I
    '''
    N = Y.shape[0]
    projected_W = np.dot(direction, W)
    alpha = 1.
    for _ in range(ls_tries):
        Y_new = np.dot(np.eye(N) + alpha * direction, Y)
        W_new = W + alpha * projected_W
        new_loss = _loss(Y_new, W_new, density)
        if new_loss < current_loss:
            return True, Y_new, W_new, new_loss, alpha * direction
        alpha /= 2.
    else:
        if verbose:
            print('line search failed, falling back to gradient')
        return False, Y_new, W_new, new_loss, alpha * direction


def _l_bfgs_direction(Y, psidY, G, s_list, y_list, r_list, lambda_min):
    q = copy(G)
    a_list = []
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        alpha = r * np.sum(s * q)
        a_list.append(alpha)
        q -= alpha * y
    z = _solve_hessian(q, Y, psidY, lambda_min)
    for s, y, r, alpha in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (alpha - beta) * s
    return -z


def _solve_hessian(G, Y, psidY, lambda_min):
    N, T = Y.shape
    # Build the diagonal of the Hessian, a.
    a = np.inner(psidY, Y ** 2) / float(T)
    # Compute the eigenvalues of the Hessian
    eigenvalues = 0.5 * (a + a.T - np.sqrt((a - a.T) ** 2 + 4.))
    # Regularize
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    a[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    # Invert the transform
    return (G * a.T - G.T) / (a * a.T - 1.)
