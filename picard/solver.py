# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

import numpy as np
import warnings
from scipy import linalg

from ._picardo import picardo
from ._picard_standard import picard_standard
from .densities import Tanh, Exp, Cube, check_density


def picard(X, fun='tanh', n_components=None, ortho=True, whiten=True,
           return_X_mean=False, max_iter=100, tol=1e-07, m=7, ls_tries=10,
           lambda_min=0.01, check_fun=True, verbose=False):
    """Perform Independent Component Analysis.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    fun : str or class, optional
        Either a built in density model ('tanh', 'exp' and 'cube'), or a custom
        density.
        A custom density is a class that should contain two methods called
        'log_lik' and 'score_and_der'. See examples in the densities.py file.


    n_components : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.

    ortho : bool, optional
        If True, uses Picard-O. Otherwise, uses the standard Picard. Picard-O
        tends to converge in fewer iterations, and finds both super Gaussian
        and sub Gaussian sources.

    whiten : boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white,
        otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.

    return_X_mean : bool, optional
        If True, X_mean is returned too.

    max_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    m : int, optional
        Size of L-BFGS's memory.

    ls_tries : int, optional
        Number of attempts during the backtracking line-search.

    lambda_min : float, optional
        Threshold on the eigenvalues of the Hessian approximation. Any
        eigenvalue below lambda_min is shifted to lambda_min.

    check_fun : bool, optionnal
        Whether to check the fun provided by the user at the beginning of
        the run. Setting it to False is not safe.

    verbose : bool, optional
        Prints informations about the state of the algorithm if True.

    Returns
    -------
    K : array, shape (n_components, n_features) | None.
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::

            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I

    Y : array, shape (n_components, n_samples) | None
        Estimated source matrix

    X_mean : array, shape (n_features,)
        The mean over features. Returned only if return_X_mean is True.
    """
    n, p = X.shape

    if not whiten and n_components is not None:
        warnings.warn('Whiten is set to false, ignoring parameter '
                      'n_components', Warning)
        n_components = None

    if n_components is None:
        n_components = min(n, p)

    # Centering the columns (ie the variables)
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, np.newaxis]
    if whiten:
        # Whitening and preprocessing by PCA
        u, d, _ = linalg.svd(X, full_matrices=False)

        del _
        K = (u / d).T[:n_components]
        del u, d
        X1 = np.dot(K, X)
        X1 *= np.sqrt(p)
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = X.astype('float')
    if fun == 'tanh':
        fun = Tanh()
    elif fun == 'exp':
        fun = Exp()
    elif fun == 'cube':
        fun = Cube()
    elif check_fun:
        check_density(fun)

    if ortho:
        Y, W, infos = picardo(X1, fun, m, max_iter, tol, lambda_min, ls_tries,
                              verbose)
    else:
        Y, W, infos = picard_standard(X1, fun, m, max_iter, tol, lambda_min,
                                      ls_tries, verbose)
    del X1
    converged = infos['converged']
    if not converged:
        gradient_norm = infos['gradient_norm']
        warnings.warn('Picard did not converge, final gradient norm = %.4g'
                      ' while the requested tolerance was %.4g. Consider'
                      'increasing the number of iterations or the tolerance.'
                      % (gradient_norm, tol))
    if not whiten:
        K = None
    if return_X_mean:
        return K, W, Y, X_mean
    else:
        return K, W, Y
