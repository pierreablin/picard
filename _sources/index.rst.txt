.. picard documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Picard
======

This is a library to run the Preconditioned ICA for Real Data (PICARD) algorithm [1]
and its orthogonal version (PICARD-O) [2]. These algorithms show fast convergence even
on real data for which sources independence do not perfectly hold.

Installation
------------

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.


conda
~~~~~

Picard can be installed with `conda-forge <https://conda-forge.org/docs/user/introduction.html>`_.
You need to add `conda-forge` to your conda channels, and then do::

  $ conda install python-picard


pip
~~~

Otherwise, to install ``picard``, you first need to install its dependencies::

	$ pip install numpy matplotlib numexpr scipy

Then install Picard with pip::

	$ pip install python-picard

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.


check
~~~~~

To check if everything worked fine, you can do::

	$ python -c 'import picard'

and it should not give any error message.

Quickstart
----------

The easiest way to get started is to copy the following lines of code
in your script:

.. code:: python

   >>> import numpy as np
   >>> from picard import picard
   >>> N, T = 3, 1000
   >>> S = np.random.laplace(size=(N, T))
   >>> A = np.random.randn(N, N)
   >>> X = np.dot(A, S)
   >>> K, W, Y = picard(X)  # doctest:+ELLIPSIS

Picard outputs the whitening matrix, `K`, the estimated unmixing matrix, `W`, and
the estimated sources `Y`. It means that:

.. math::

    Y = W K X


NEW in 0.6: scikit-learn compatible API
---------------------------------------

Introducing `picard.Picard`, which mimics `sklearn.decomposition.FastICA` behavior:

.. code:: python

    >>> from sklearn.datasets import load_digits
    >>> from picard import Picard
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = Picard(n_components=7)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape


Dependencies
------------

These are the dependencies to use Picard:

* numpy (>=1.8)
* matplotlib (>=1.3)
* numexpr (>= 2.0)
* scipy (>=0.19)
* scikit-learn (>=0.23)


These are the dependencies to run the EEG example:

* mne (>=0.14)

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, Alexandre Gramfort
    Faster independent component analysis by preconditioning with Hessian approximations
    IEEE Transactions on Signal Processing, 2018
    https://arxiv.org/abs/1706.08171

    Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
    Faster ICA under orthogonal constraint
    ICASSP, 2018
    https://arxiv.org/abs/1711.10873


Bug reports
-----------

Use the `github issue tracker <https://github.com/pierreablin/picard/issues>`_ to report bugs.


API
---

.. toctree::
    :maxdepth: 1

    api.rst
