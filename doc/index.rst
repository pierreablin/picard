.. lbfgsica documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lbfgsica
==========

This is a library to run a faster ICA solver based on L-BFGS combined a preconditioning strategy.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``lbfgsica``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy numexpr

Then install lbfgsica::

	$ pip install git+https://github.com/pierreablin/l-bfgs-ica.git#egg=lbfgsica

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import lbfgsica'

and it should not give any error messages.

Quickstart
==========

The easiest way to get started is to copy the following three lines of code
in your script:

.. code:: python

	>>> from lbfgsica import run_ica
	>>> run_ica(...)


Bug reports
===========

Use the `github issue tracker <https://github.com/pierreablin/l-bfgs-ica/issues>`_ to report bugs.

Cite
====

[1] Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
    Faster ICA by preconditioning with Hessian approximations
    ArXiv Preprint, June 2017
