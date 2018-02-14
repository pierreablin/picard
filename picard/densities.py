import numpy as np
import numexpr as ne

from itertools import cycle
from scipy.optimize import check_grad
from numpy.testing import assert_allclose


class density:
    def score_and_der(self, Y):
        return self.score(Y), self.score_der(Y)

    def check(self, tol=1e-6, n_test=10, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        Y = rng.randn(n_test)

        def score2(Y):
            return self.score_and_der(Y)[0]

        def score_der2(Y):
            return self.score_and_der(Y)[1]

        der_list = [self.score, self.score_der, score2, score_der2]
        err_msgs = ['score', 'score derivative', 'score in score_and_der',
                    'score derivative in score_and_der']
        for f, fprime, err_msg in zip(cycle([self.log_lik, self.score]),
                                      der_list, err_msgs):
            for y in Y:
                err = check_grad(f, fprime, np.array([y]))
                assert_allclose(err, 0, atol=tol, rtol=0,
                                err_msg='Wrong %s' % err_msg)


class tanh(density):
    def __init__(self, params=None):
        self.alpha = 1.
        if params is not None:
            if 'alpha' in params:
                self.alpha = params['alpha']

    def log_lik(self, Y):
        alpha = self.alpha  # noqa
        return ne.evaluate('abs(Y) + log1p(exp(-2. * alpha * abs(Y))) / alpha')

    def score(self, Y):
        alpha = self.alpha  # noqa
        return ne.evaluate('tanh(alpha * Y)')

    def score_der(self, Y):
        alpha = self.alpha  # noqa
        return ne.evaluate('alpha - alpha * tanh(alpha * Y) ** 2')

    def score_and_der(self, Y):
        alpha = self.alpha
        score = ne.evaluate('tanh(alpha * Y)')
        return score, alpha - alpha * score ** 2


class exp(density):
    def __init__(self, params=None):
        self.alpha = 1.
        if params is not None:
            if 'alpha' in params:
                self.alpha = params['alpha']

    def log_lik(self, Y):
        a = self.alpha  # noqa
        return ne.evaluate('-exp(- a * Y ** 2 / 2.) / a')

    def score(self, Y):
        a = self.alpha  # noqa
        return ne.evaluate('Y * exp(- a * Y ** 2 / 2.)')

    def score_der(self, Y):
        a = self.alpha  # noqa
        return ne.evaluate('(1 - a * Y ** 2) * exp(- a * Y ** 2 / 2.)')

    def score_and_der(self, Y):
        a = self.alpha  # noqa
        Y_sq = ne.evaluate('Y ** 2')  # noqa
        K = ne.evaluate('exp(- a / 2. * Y_sq)')  # noqa
        return ne.evaluate('Y * K'), ne.evaluate('(1- a * Y_sq) * K')


class cube(density):
    def log_lik(self, Y):
        return ne.evaluate('Y ** 4 / 4')

    def score(self, Y):
        return ne.evaluate('Y ** 3')

    def score_der(self, Y):
        return ne.evaluate('3 * Y ** 2')
