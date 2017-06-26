import numpy as np
from lbfgsica import lbfgs_ica


def test_lbgfs_solver():
    N, T = 2, 1000
    S = np.random.laplace(size=(N, T))
    A = np.random.randn(N, N)
    X = np.dot(A, S)
    lbfgs_ica(X, verbose=True)


if __name__ == '__main__':
    test_lbgfs_solver()
