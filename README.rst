L-BFGS ICA
==========

|CircleCI|_ |Travis|_ |Codecov|_

.. |CircleCI| image:: https://circleci.com/gh/lbfgsica/lbfgsica/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/lbfgsica/lbfgsica

.. |Travis| image:: https://api.travis-ci.org/lbfgsica/lbfgsica.svg?branch=master
.. _Travis: https://travis-ci.org/lbfgsica/lbfgsica

.. |Codecov| image:: http://codecov.io/github/lbfgsica/lbfgsica/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/lbfgsica/lbfgsica?branch=master

This repository hosts code to run a faster ICA solver based on L-BFGS
combined a preconditioning strategy.

Dependencies
------------

These are the dependencies to use lbfgsica:

* numpy (>=1.8)
* matplotlib (>=1.3)
* scipy (>=0.16)
* numexpr (>= 2.0) 

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
    Faster ICA by preconditioning with Hessian approximations
    ArXiv Preprint, June 2017
