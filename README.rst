====
lnpy
====


.. image:: https://img.shields.io/pypi/v/lnpy.svg
        :target: https://pypi.python.org/pypi/lnpy

.. image:: https://img.shields.io/travis/wpk-nist-gov/lnpy.svg
        :target: https://travis-ci.com/wpk-nist-gov/lnpy

.. image:: https://readthedocs.org/projects/lnpy/badge/?version=latest
        :target: https://lnpy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Package to analyze :math:`\ln \Pi(N)` data from Transition Matrix Monte Carlo simulation


Installation
============

From Source
-----------

.. code-block:: console

   # clone repo
   git clone {repo}
   cd {repo}

   # create needed environment
   conda env create -n {optional-name] -f environment.yaml

   # activate environment
   conda activate {optional-name/lnpy-env (default)}

   # install in development mode
   pip install -e . --no-deps

   # Optionally run tests.  This requires pytest
   conda install pytest

   pytest -x -v



Quick Intro
===========

Take a look at the `basic usage`_ notebook for a quick introduction.


Licence
=======

.. include:: LICENCE



Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`basic usage`:_ docs/notebooks/Basic_usage.ipynb
