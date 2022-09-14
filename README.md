# tmmc-lnpy

A package to analyze $\ln \Pi(N)$ data from Transition Matrix Monte Carlo
simulation.  The main output from TMMC simulations, $\ln \Pi(N)$, provides a means to calculate a host of thermodynamic
properties.  Moreover, if $\ln \Pi(N)$ is calculated at a specific chemical potential, it can be reweighted to provide
thermodynamic information at a different chemical potential

``tmmc-lnpy`` provides a wide array of routines to analyze $\ln \Pi(N)$.  These include:

* Reweighting to arbitrary chemical potential
* Segmenting $\ln \Pi(N)$ (to identify unique phases)
* Containers for interacting with several values of $\ln \Pi(N)$ in a vectorized way.
* Calculating thermodynamic properties from these containers
* Calculating limits of stability, and phase equilibrium

# Status

``tmmc-lnpy`` is actively used by it's author.  Pull requests are welcome!

# Installation

``` console
# from pypi
pip install tmmc-lnpy

# from conda
conda install -c wpk-nist tmmc-lnpy

# from source
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

# Testing

Testing status is mostly regression tests right now.  Further test modules will be added in the near future.


# Getting started

Take a look at the [basic usage](docs/src/notebooks/Basic_usage.ipynb)
notebook for a quick introduction.


Note that the distribution name ``tmmc-lnpy`` is different than the import name ``lnpy`` due to name clashing on pypi.

``` python
import lnpy
import lnpy.examples

ref = lnpy.examples.load_example_maskddata('lj_sub')
```

# License

See [LICENSE](LICENSE)


# Related work

Please checkout [feasst](https://github.com/usnistgov/feasst), a TMMC simulation package.  We hope to create routines to more
directly interact with feasst output in the near future.

# TODO

- [ ] More documentation/examples
- [ ] Update Spinodal/Binodal accessor api
- [ ] Typing
- [ ] Interface to thermoextrap
- [ ] Most of the testing is regression testing. Should add some unit tests as well.
- [ ] Strip out unused legacy code.


# Contact

The author can be reached at wpk@nist.gov


# Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
