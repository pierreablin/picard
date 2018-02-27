import numpy as np
import numexpr as ne

from scipy.optimize import check_grad
from numpy.testing import assert_allclose


class Density(object):
    """Interface to use custom densities for Picard

    These elements can be passed in the ``fun`` argument of Picard.

    It must implement a method `̀ log_lik`` which evaluates the
    log-likelihood for the samples.
    It must also contain either two methods ``score`` and ``score_der``
    which respectively return the score and its derivative,
    or a method ``score_and_der`` which returns both in a tuple of numpy
    arrays. In many cases, computing the score and its derivative at the same
    time can save some computations (see the example).
    Parameters
    ----------
    log_lik : callable f(Y)
        Returns an array containing the log-likelihood for each sample in Y.
    score : callable f(Y)
        Returns an array containing the score for each sample in Y.
    score_der : callable f(Y)
        Returns an array containing the derivative of the score for each sample
        in Y.
    score_and_der : callable f(Y)
        Returns a tuple of arrays (psiY, psidY), where psiY contains the score
        for each sample of Y and psidY contains the derivative of the score for
        each sample in Y.
    Examples
    --------
    >>> import numpy as np
    >>> from picard import Density
    >>> def log_lik(Y):
    ...     return Y ** 4 / 4
    ...
    >>> def score(Y):
    ...     return Y ** 3
    ...
    >>> def score_der(Y):
    ...     return 3 * Y ** 2
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])
    """
    def score_and_der(self, Y):
        return self.score(Y), self.score_der(Y)


def check_density(density, tol=1e-6, n_test=10, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    Y = rng.randn(n_test)

    def score(Y):
        return density.score_and_der(Y)[0]

    def score_der(Y):
        return density.score_and_der(Y)[1]

    err_msgs = ['score', 'score derivative']
    for f, fprime, err_msg in zip([density.log_lik, score], [score, score_der],
                                  err_msgs):
        for y in Y:
            err = check_grad(f, fprime, np.array([y]))
            assert_allclose(err, 0, atol=tol, rtol=0,
                            err_msg='Wrong %s' % err_msg)


class Tanh(object):
    def __init__(self, params=None):
        self.alpha = 1.
        if params is not None:
            if 'alpha' in params:
                self.alpha = params['alpha']

    def log_lik(self, Y):
        alpha = self.alpha  # noqa
        return ne.evaluate('abs(Y) + log1p(exp(-2. * alpha * abs(Y))) / alpha')

    def score_and_der(self, Y):
        alpha = self.alpha
        score = ne.evaluate('tanh(alpha * Y)')
        return score, alpha - alpha * score ** 2


class Exp(object):
    def __init__(self, params=None):
        self.alpha = 1.
        if params is not None:
            if 'alpha' in params:
                self.alpha = params['alpha']

    def log_lik(self, Y):
        a = self.alpha  # noqa
        return ne.evaluate('-exp(- a * Y ** 2 / 2.) / a')

    def score_and_der(self, Y):
        a = self.alpha  # noqa
        Y_sq = ne.evaluate('Y ** 2')  # noqa
        K = ne.evaluate('exp(- a / 2. * Y_sq)')  # noqa
        return ne.evaluate('Y * K'), ne.evaluate('(1- a * Y_sq) * K')


class Cube(object):
    def log_lik(self, Y):
        return ne.evaluate('Y ** 4 / 4')

    def score_and_der(self, Y):
        return ne.evaluate('Y ** 3'), ne.evaluate('3 * Y ** 2')
