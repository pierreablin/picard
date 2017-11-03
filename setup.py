#! /usr/bin/env python

import os
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Preconditoned ICA for Real Data"""

DISTNAME = 'picard'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierre.ablin@inria.fr'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/pierreablin/picard.git'
VERSION = '0.1.dev0'


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=package_tree('picard'),
          )
