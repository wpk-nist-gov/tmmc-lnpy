[![Repo][repo-badge]][repo-link]
[![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: black][black-badge]][black-link]


[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/ambv/black
[pypi-badge]: https://img.shields.io/pypi/v/tmmc-lnpy
<!-- [pypi-badge]: https://badge.fury.io/py/tmmc-lnpy -->
[pypi-link]: https://pypi.org/project/tmmc-lnpy
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/tmmc-lnpy/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/tmmc-lnpy
[conda-badge]: https://img.shields.io/conda/v/wpk-nist/tmmc-lnpy
[conda-link]: https://anaconda.org/wpk-nist/tmmc-lnpy
<!-- Use total link so works from anywhere -->
[license-badge]: https://img.shields.io/pypi/l/cmomy?color=informational
[license-link]: https://github.com/usnistgov/tmmc-lnpy/blob/master/LICENSE
<!-- For more badges, see https://shields.io/category/other and https://naereen.github.io/badges/ -->

[numpy]: https://numpy.org
[Numba]: https://numba.pydata.org/
[xarray]: https://docs.xarray.dev/en/stable/


# `tmmc-lnpy`

## Overview

A package to analyze $\ln \Pi(N)$ data from Transition Matrix Monte Carlo
simulation.  The main output from TMMC simulations, $\ln \Pi(N)$, provides a means to calculate a host of thermodynamic
properties.  Moreover, if $\ln \Pi(N)$ is calculated at a specific chemical potential, it can be reweighted to provide
thermodynamic information at a different chemical potential




## Features

``tmmc-lnpy`` provides a wide array of routines to analyze $\ln \Pi(N)$.  These include:

* Reweighting to arbitrary chemical potential
* Segmenting $\ln \Pi(N)$ (to identify unique phases)
* Containers for interacting with several values of $\ln \Pi(N)$ in a vectorized way.
* Calculating thermodynamic properties from these containers
* Calculating limits of stability, and phase equilibrium


## Status

This package is actively used by the author.  Please feel free to create a pull request for wanted features and suggestions!


## Quick start

Use one of the following

``` bash
pip install tmmc-lnpy
```

or

``` bash
conda install -c wpk-nist tmmc-lnpy
```

## Example usage

Note that the distribution name ``tmmc-lnpy`` is different than the import name ``lnpy`` due to name clashing on pypi.

``` python
import lnpy
import lnpy.examples

ref = lnpy.examples.load_example_maskddata('lj_sub')
```


<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for a look at `tmmc-lnpy` in action.

## License

This is free software.  See [LICENSE][license-link].

## Related work

This package is used for with [thermoextrap](https://github.com/usnistgov/thermo-extrap) to analyze thermodynamically extrapolated macro state probability distributions.

## Contact

The author can be reached at wpk@nist.gov.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
