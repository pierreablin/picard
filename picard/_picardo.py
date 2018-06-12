# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy.linalg import expm

from ._tools import (gradient, proj_hessian_approx, regularize_hessian,
                     l_bfgs_direction, line_search)

from .densities import Tanh


def picardo(X, density=Tanh(), m=7, maxiter=100, tol=1e-9, lambda_min=0.01,
            ls_tries=10, verbose=False):
    '''Runs the Picard-O algorithm


    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the signals that have to be unmixed. N is the
        number of signals, T is the number of samples. X has to be centered and
        white.

    m : int
        Size of L-BFGS's memory. Typical values for m are in the range 3-15

    maxiter : int
        Maximal number of iterations for the algorithm

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the projected gradient gets smaller than tol.

    lambda_min : float
        Constant used to regularize the Hessian approximation. The
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
    current_loss = None
    sign_change = False
    requested_tolerance = False
    gradient_norm = 1.
    for n in range(maxiter):
        # Compute the score function
        psiY, psidY = density.score_and_der(Y)
        psidY_mean = np.mean(psidY, axis=1)
        del psidY
        # Compute the relative gradient
        g = gradient(Y, psiY)
        del psiY
        # Compute the signs of the kurtosis
        K = psidY_mean - np.diag(g)
        signs = np.sign(K)
        if n > 0:
            sign_change = np.any(signs != old_signs)  # noqa
        old_signs = signs  # noqa
        # Update the gradient
        g *= signs[:, None]
        psidY_mean *= signs
        # Project
        G = (g - g.T) / 2.
        # Stopping criterion
        gradient_norm = np.max(np.abs(G))
        if gradient_norm < tol:
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
        # Flush the memory if there is a sign change.
        if sign_change:
            current_loss = None
            s_list, y_list, r_list = [], [], []
        # Compute the Hessian approximation and regularize
        h = proj_hessian_approx(Y, psidY_mean, g)
        h = regularize_hessian(h, lambda_min)
        # Find the L-BFGS direction
        direction = l_bfgs_direction(G, h, s_list, y_list, r_list)
        # Do a line_search in that direction :
        converged, new_Y, new_loss, alpha =\
            line_search(Y, signs, density, direction, current_loss, ls_tries)
        # If the line search fails, restart in the gradient direction
        if not converged:
            direction = -G
            s_list, y_list, r_list = [], [], []
            _, new_Y, new_loss, alpha =\
                line_search(Y, signs, density, direction, current_loss,
                            ls_tries)
        direction *= alpha
        Y = new_Y
        W = expm(direction).dot(W)
        current_loss = new_loss
        if verbose:
            print('iteration %d, gradient norm = %.4g' % (n, gradient_norm))
    infos = dict(converged=requested_tolerance, gradient_norm=gradient_norm,
                 n_iterations=n)
    return Y, W, infos
