<!-- markdownlint-disable MD041 -->

[![Repo][repo-badge]][repo-link] [![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: black][black-badge]][black-link]

<!--
  For more badges, see
  https://shields.io/category/other
  https://naereen.github.io/badges/
  [pypi-badge]: https://badge.fury.io/py/tmmc-lnpy
-->

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[pypi-badge]: https://img.shields.io/pypi/v/tmmc-lnpy
[pypi-link]: https://pypi.org/project/tmmc-lnpy
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/tmmc-lnpy/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/tmmc-lnpy
[conda-badge]: https://img.shields.io/conda/v/wpk-nist/tmmc-lnpy
[conda-link]: https://anaconda.org/wpk-nist/tmmc-lnpy
[license-badge]: https://img.shields.io/pypi/l/cmomy?color=informational
[license-link]: https://github.com/usnistgov/tmmc-lnpy/blob/main/LICENSE

<!-- other links -->

# `tmmc-lnpy`

## Overview

A package to analyze $\ln \Pi(N)$ data from Transition Matrix Monte Carlo
simulation. The main output from TMMC simulations, $\ln \Pi(N)$, provides a
means to calculate a host of thermodynamic properties. Moreover, if $\ln \Pi(N)$
is calculated at a specific chemical potential, it can be reweighted to provide
thermodynamic information at a different chemical potential

## Features

`tmmc-lnpy` provides a wide array of routines to analyze $\ln \Pi(N)$. These
include:

- Reweighting to arbitrary chemical potential
- Segmenting $\ln \Pi(N)$ (to identify unique phases)
- Containers for interacting with several values of $\ln \Pi(N)$ in a vectorized
  way.
- Calculating thermodynamic properties from these containers
- Calculating limits of stability, and phase equilibrium

## Status

This package is actively used by the author. Please feel free to create a pull
request for wanted features and suggestions!

## Quick start

Use one of the following

```bash
pip install tmmc-lnpy
```

or

```bash
conda install -c wpk-nist tmmc-lnpy
```

## Example usage

Note that the distribution name `tmmc-lnpy` is different than the import name
`lnpy` due to name clashing on pypi.

```python
>>> import numpy as np
>>> import lnpy
>>> import lnpy.examples

>>> ref = lnpy.examples.load_example_lnpimasked('lj_sub')

>>> phase_creator = lnpy.PhaseCreator(nmax=1, ref=ref)
>>> build_phases = phase_creator.build_phases_mu([None])
>>> collection = lnpy.lnPiCollection.from_builder(
...     lnzs=np.linspace(-10, 3, 5), build_phases=build_phases
... )


# Collections are like pandas.Series
>>> collection
<class lnPiCollection>
lnz_0   phase
-10.00  0        [-10.0]
-6.75   0        [-6.75]
-3.50   0         [-3.5]
-0.25   0        [-0.25]
 3.00   0          [3.0]
dtype: object


# Access xarray backend for Grand Canonical properties with `xge` accessor
>>> collection.xge.betaOmega()
<xarray.DataArray 'betaOmega' (lnz_0: 5, phase: 1)>
array([[-2.32445630e-02],
       [-6.03695807e-01],
       [-1.85523371e+02],
       [-1.54471391e+03],
       [-2.95801694e+03]])
Coordinates:
  * lnz_0    (lnz_0) float64 -10.0 -6.75 -3.5 -0.25 3.0
  * phase    (phase) int64 0
    beta     float64 1.372
    volume   float64 512.0
Attributes:
    dims_n:         ['n_0']
    dims_lnz:       ['lnz_0']
    dims_comp:      ['component']
    dims_state:     ['lnz_0', 'beta', 'volume']
    dims_rec:       ['sample']
    standard_name:  grand_potential
    long_name:      $\beta \Omega(\mu,V,T)$


```

<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for a look at `tmmc-lnpy` in action.

## License

This is free software. See [LICENSE][license-link].

## Related work

This package is used for with
[thermoextrap](https://github.com/usnistgov/thermo-extrap) to analyze
thermodynamically extrapolated macro state probability distributions.

## Contact

The author can be reached at wpk@nist.gov.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
