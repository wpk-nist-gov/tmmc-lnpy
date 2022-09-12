lnpy
====

Package to analyze :math:`\ln \Pi(N)` data from Transition Matrix Monte
Carlo simulation

Installation
------------

From Source
~~~~~~~~~~~

.. code:: console

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


Credits
~~~~~~~

This package was created with
`Cookiecutter <https://github.com/audreyr/cookiecutter>`__ and the
`wpk-nist-gov/cookiecutter-pypackage <https://github.com/wpk-nist-gov/cookiecutter-pypackage>`__
Project template forked from
`audreyr/cookiecutter-pypackage <https://github.com/audreyr/cookiecutter-pypackage>`__.
