L-BFGS ICA
==========

|Travis|_ |Codecov|_

.. |Travis| image:: https://api.travis-ci.org/pierreablin/l-bfgs-ica.svg?branch=master
.. _Travis: https://travis-ci.org/pierreablin/l-bfgs-ica

.. |Codecov| image:: http://codecov.io/github/pierreablin/l-bfgs-ica/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/pierreablin/l-bfgs-ica?branch=master

This repository hosts code to run a faster ICA solver based on L-BFGS
combined with a preconditioning strategy.

Dependencies
------------

These are the dependencies to use lbfgsica:

* numpy (>=1.8)
* matplotlib (>=1.3)
* scipy (>=0.16)
* numexpr (>= 2.0) 

These are the dependencies to run the EEG example:

* mne (>=0.14)
* scikit-learn (>=0.18)

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
    Faster ICA by preconditioning with Hessian approximations
    ArXiv Preprint, June 2017
