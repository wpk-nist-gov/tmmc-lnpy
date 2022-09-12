# lnpy

[![image](https://img.shields.io/pypi/v/lnpy.svg)](https://pypi.python.org/pypi/lnpy)

<!-- [![Documentation Status](https://readthedocs.org/projects/lnpy/badge/?version=latest)](https://lnpy.readthedocs.io/en/latest/?badge=latest) -->

Package to analyze $\ln \Pi(N)$ data from Transition Matrix Monte Carlo
simulation

## Installation

### From Source

``` console
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
```

## Quick Intro

Take a look at the [basic usage](docs/notebooks/Basic_usage.ipynb)
notebook for a quick introduction.

## License

See [LICENSE](LICENSE)

## TODO

[ ] More documentation/examples
[ ] Update Spinodal/Binodal accessor api
[ ] Interface to thermoextrap

### Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
