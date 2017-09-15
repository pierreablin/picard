Picard : Preconditioned ICA for Real Data
==========

|Travis|_ |Codecov|_

.. |Travis| image:: https://api.travis-ci.org/pierreablin/picard.svg?branch=master
.. _Travis: https://travis-ci.org/pierreablin/picard

.. |Codecov| image:: http://codecov.io/github/pierreablin/picard/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/pierreablin/picard?branch=master

This repository hosts code of the Preconditioned ICA for Real Data (Picard) algorithm.

See the `documentation <https://pierreablin.github.io/picard/index.html>`_.


Algorithm
---------

Picard solves the maximum likelihood formulation of independent component analysis.
It uses a preconditioned L-BFGS strategy, resulting in a very fast convergence.


Installation
------------

To install Picard, the simplest way is to use pip::

	$ pip install git+https://github.com/pierreablin/picard.git#egg=picard


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
   >>> Y, W = picard(X)

Picard outputs the estimated sources, Y, and estimated unmixing matrix, W.


Dependencies
------------

These are the dependencies to use Picard:

* numpy (>=1.8)
* matplotlib (>=1.3)
* numexpr (>= 2.0)

These are the dependencies to run the EEG example:

* mne (>=0.14)
* scikit-learn (>=0.18)
* scipy (>=0.19)

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
    Faster independent component analysis by preconditioning with Hessian approximations
    ArXiv Preprint, June 2017
