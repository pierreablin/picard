# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Francois Cardoso <cardoso@iap.fr>
#
# License: BSD (3-clause)

import numpy as np
try:
    import numexpr as ne
except ImportError:
    ne = None
from scipy.optimize import check_grad
from numpy.testing import assert_allclose


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
        if ne is None:
            absY = np.abs(Y)
            np.exp(-2. * alpha * absY)
            return absY + np.log1p(np.exp(-2. * alpha * absY)) / alpha
        return ne.evaluate('abs(Y) + log1p(exp(-2. * alpha * abs(Y))) / alpha')

    def score_and_der(self, Y):
        alpha = self.alpha
        if ne is None:
            score = np.tanh(alpha * Y)
        else:
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
        if ne is None:
            return -np.exp(- a * Y ** 2 / 2.) / a
        return ne.evaluate('-exp(- a * Y ** 2 / 2.) / a')

    def score_and_der(self, Y):
        a = self.alpha  # noqa
        if ne is None:
            Y_sq = Y ** 2
            K = np.exp(- a / 2. * Y_sq)
            return Y * K, (1 - a * Y_sq) * K
        else:
            Y_sq = ne.evaluate('Y ** 2')  # noqa
            K = ne.evaluate('exp(- a / 2. * Y_sq)')  # noqa
            return ne.evaluate('Y * K'), ne.evaluate('(1- a * Y_sq) * K')


class Cube(object):
    def log_lik(self, Y):
        if ne is None:
            return Y ** 4 / 4
        return ne.evaluate('Y ** 4 / 4')

    def score_and_der(self, Y):
        if ne is None:
            return Y ** 3, 3 * Y ** 2
        return ne.evaluate('Y ** 3'), ne.evaluate('3 * Y ** 2')
