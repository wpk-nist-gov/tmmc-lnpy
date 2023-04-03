# tmmc-lnpy

Package to analyze $\ln \Pi(N)$ data from Transition Matrix Monte
Carlo simulation.

## Links

- [Github repo](https://github.com/usnistgov/tmmc-lnpy)
- [Documentation](https://pages.nist.gov/tmmc-lnpy/)

## Installation

### From Source

```console
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
```

## Quick Intro

Take a look at the [basic usage](https://github.com/usnistgov/tmmc-lnpy/blob/master/docs/notebooks/Basic_usage.ipynb)
notebook for a quick introduction.

Note that the distribution name `tmmc-lnpy` is different than the package name `lnpy`, due to name conflicts on pypi.  To load the package in python, do the following:

```python
import lnpy
import lnpy.examples

ref = lnpy.examples("lj_sup")
```

### Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
