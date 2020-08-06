"""
==================================
Using a custom density with Picard
==================================

This example shows how to use custom densities using Picard

"""  # noqa

# Author: Pierre Ablin <pierre.ablin@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from picard import picard, permute

print(__doc__)

###############################################################################
# Build a custom density where the score function is x + tanh(x)


class CustomDensity(object):
    def log_lik(self, Y):
        return Y ** 2 / 2 + np.log(np.cosh(Y))

    def score_and_der(self, Y):
        tanhY = np.tanh(Y)
        return Y + tanhY, 2 - tanhY ** 2


custom_density = CustomDensity()

###############################################################################
# Plot the corresponding functions

x = np.linspace(-2, 2, 100)
log_likelihood = custom_density.log_lik(x)
psi, psi_der = custom_density.score_and_der(x)

names = ['log-likelihood', 'score', 'score derivative']

plt.figure()
for values, name in zip([log_likelihood, psi, psi_der], names):
    plt.plot(x, values, label=name)
plt.legend()
plt.title("Custom density")
plt.show()

###############################################################################
# Run Picard on toy dataset using this density

rng = np.random.RandomState(0)
N, T = 5, 1000
S = rng.laplace(size=(N, T))
A = rng.randn(N, N)
X = np.dot(A, S)
K, W, Y = picard(X, fun=custom_density, random_state=0)
plt.figure()
plt.imshow(permute(W.dot(K).dot(A)), interpolation='nearest')
plt.title('Product between the estimated unmixing matrix and the mixing'
          'matrix')
plt.show()
