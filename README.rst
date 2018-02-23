Picard : Preconditioned ICA for Real Data
=========================================

|Travis|_ |Codecov|_

.. |Travis| image:: https://api.travis-ci.org/pierreablin/picard.svg?branch=master
.. _Travis: https://travis-ci.org/pierreablin/picard

.. |Codecov| image:: http://codecov.io/github/pierreablin/picard/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/pierreablin/picard?branch=master

This repository hosts Python/Octave/Matlab code of the Preconditioned ICA
for Real Data (Picard) and Picard-O algorithms.

See the `documentation <https://pierreablin.github.io/picard/index.html>`_.

Algorithm
---------

Picard is an algorithm for maximum likelihood independent component analysis.
It solves the same problem as Infomax, faster.
It uses a preconditioned L-BFGS strategy, resulting in a very fast convergence.


Picard-O uses an adaptation of that strategy to solve the same problem under the
constraint of whiteness of the signals. It solves the same problem as
FastICA, but faster.

Picard-O is able to recover both super-Gaussian and sub-Gaussian sources.


Installation
------------

To install the package, the simplest way is to use pip to get the latest release::

  $ pip install python-picard

or to get the latest version of the code::

  $ pip install git+https://github.com/pierreablin/picard.git#egg=picard

The Matlab/Octave version of Picard and Picard-O is `available here <https://github.com/pierreablin/picard/tree/master/matlab_octave>`_.

Quickstart
----------

To get started, you can build a synthetic signals matrix:

.. code:: python

   >>> import numpy as np
   >>> N, T = 3, 1000
   >>> S = np.random.laplace(size=(N, T))
   >>> A = np.random.randn(N, N)
   >>> X = np.dot(A, S)

And then feed Picard with it:

.. code:: python

   >>> from picard import picard
   >>> K, W, Y = picard(X)

Picard outputs the whitening matrix, K, the estimated unmixing matrix, W, and
the estimated sources Y. It means that:

.. math::

    Y = W K X

Dependencies
------------

These are the dependencies to use Picard:

* numpy (>=1.8)
* matplotlib (>=1.3)
* numexpr (>= 2.0)
* scipy (>=0.19)


These are the dependencies to run the EEG example:

* mne (>=0.14)

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, Alexandre Gramfort
    Faster independent component analysis by preconditioning with Hessian approximations
    ArXiv Preprint, June 2017
    https://arxiv.org/abs/1706.08171

    Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
    Faster ICA under orthogonal constraint
    ArXiv Preprint, Nov 2017
    https://arxiv.org/abs/1711.10873
