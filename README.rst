tmmc-lnpy
=========

Package to analyze :math:`\ln \Pi(N)` data from Transition Matrix Monte
Carlo simulation.

Installation
------------

From Source
~~~~~~~~~~~

.. code:: console

   # * From pip
   pip install tmmc-lnpy

   # * From conda/mamba
   conda install -c wpk-nist tmmc-lnpy

   # * From Source
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
-----------

Take a look at the `basic usage <https://github.com/wpk-nist-gov/tmmc-lnpy/blob/master/docs/notebooks/Basic_usage.ipynb>`__
notebook for a quick introduction.

Note that the distrubution name `tmmc-lnpy` is different than the package name `lnpy`, due to name conflicts on pypi.  To load the package in python, do the following:

.. code:: python

    import lnpy
    import lnpy.examples

    ref = lnpy.examples("lj_sup")


Credits
~~~~~~~

This package was created with
`Cookiecutter <https://github.com/audreyr/cookiecutter>`__ and the
`wpk-nist-gov/cookiecutter-pypackage <https://github.com/wpk-nist-gov/cookiecutter-pypackage>`__
Project template forked from
`audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`__.
