import numpy as np
import numexpr as ne
from copy import deepcopy


def lbfgs_ica(X, m=7, maxiter=100, precon=1, tol=1e-7, lambda_min=0.01,
              ls_tries=5, verbose=False):
    '''
    Runs the L-BFGS ICA algorithm as described in "Faster ICA by
    preconditioning with Hessian approximations"

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

    Returns:
    --------
    Y : array, shape (N, T)
        The estimated source matrix

    W : array, shape (N, N)
        The estimated unmixing matrix, such that Y = WX.
    '''
    # Init
    mode = 'ne'
    N, T = X.shape
    W = np.eye(N)
    Y = deepcopy(X)
    s_list = []
    y_list = []
    G_old = 0.
    for n in range(maxiter):
        # Compute the score function
        thY = tanh(Y, mode)
        # Compute the relative gradient
        G = np.inner(thY, Y) / float(T) - np.eye(N)
        # Stopping criterion
        G_norm = np.max(np.abs(G))
        if G_norm < tol:
            break
        # Find the L-BFGS direction
        # direction = L_BFGS_direction(Y, thY, G, s_list, y_list, precon,
        #                              lambda_min, mode)
        direction = solveH(G, Y, thY, precon, lambda_min, mode)
        # Do a line_search in that direction:
        if n == 0:
            current_loss = loss(Y, W, mode)
        new_Y, new_W, new_loss = line_search(Y, W, direction, current_loss,
                                             ls_tries, mode)
        if new_Y is None:
            direction = G
            new_Y, new_W, new_loss = line_search(Y, W, direction, current_loss,
                                                 15, mode)
        Y = new_Y
        W = new_W
        current_loss = new_loss
        # Update the memory
        if n > 0:
            s_list.append(direction)
            y_list.append(G - G_old)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        G_old = G
        if verbose:
            print('iteration %d, loss = %.4g, gradient norm = %.2g' %
                  (n, current_loss, G_norm))
    return Y, W


def tanh(Y, mode):
    '''
    Computes tanh(Y / 2.), using numexpr if available, and numpy otherwise
    '''
    if mode == 'ne':
        return ne.evaluate('tanh(Y / 2.)') # noqa
    else:
        return np.tanh(Y / 2.)


def loss(Y, W, mode):
    '''
    Computes the loss function for Y, W, using numexpr if available and numpy
    otherwise
    '''
    T = Y.shape[1]
    log_det = np.linalg.slogdet(W)[1]
    if mode == 'ne':
        absY = ne.evaluate('abs(Y)') # noqa
        logcoshY = ne.evaluate('sum(absY + 2. * log1p(exp(-absY)))') # noqa
    else:
        absY = np.abs(Y)
        logcoshY = np.sum(absY + 2. * np.log1p(np.exp(-absY)))
    return - log_det + logcoshY / float(T)


def line_search(Y, W, direction, current_loss, ls_tries, mode):
    projected_Y = np.dot(direction, Y)
    projected_W = np.dot(direction, W)
    alpha = 1.
    for _ in range(ls_tries):
        Y_new = Y - alpha * projected_Y
        W_new = W - alpha * projected_W
        new_loss = loss(Y_new, W_new, mode)
        if new_loss < current_loss:
            # print alpha
            return Y_new, W_new, new_loss
        alpha /= 2.
    else:
        print('ls fail')
        return None, None, None


def L_BFGS_direction(Y, thY, G, s_list, y_list, precon, lambda_min, mode):
    q = G
    r_list = []
    a_list = []
    for s, y in zip(s_list, y_list):
        r_list.append(1. / (np.sum(s * y)))
    for s, y, r in zip(reversed(s_list), reversed(y_list), reversed(r_list)):
        a = r * np.sum(s * q)
        a_list.append(a)
        q -= a * y
    z = solveH(q, Y, thY, precon, lambda_min, mode)
    for s, y, r, a in zip(s_list, y_list, r_list, reversed(a_list)):
        beta = r * np.sum(y * z)
        z += (a - beta) * s
    return z


def solveH(G, Y, thY, precon, lambda_min, mode):
    N, T = Y.shape
    # Compute the derivative of the score
    if mode == 'np':
        psidY = (- np.square(thY) + 1.) / 2.
    else:
        psidY = ne.evaluate('(- thY ** 2 + 1.) / 2.') # noqa
    # Build the diagonal of the Hessian, a.
    if precon == 2:
        a = np.inner(psidY, Y ** 2) / float(T) + np.eye(N)
    else:
        Y_squared = Y ** 2
        sigma2 = np.mean(Y_squared, axis=1)
        psidY_mean = np.mean(psidY, axis=1)
        a = psidY_mean[:, None] * sigma2[None, :]
        diagonal_term = np.mean(Y_squared * psidY) + 1.
        a[np.diag_indices_from(a)] = diagonal_term
    # Compute the eigenvalues of the Hessian
    eigenvalues = 0.5 * (a + a.T + np.sqrt((a-a.T) ** 2 + 4.))
    # Regularize
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    a[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    # Invert the transform
    return (G * a.T - G.T) / (a * a.T - 1.)


if __name__ == '__main__':
    N, T = 2, 1000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    X = np.dot(A, S)
    lbfgs_ica(X, verbose=True)
