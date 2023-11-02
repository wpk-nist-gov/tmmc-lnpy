from pytest import approx

import numpy as np

from lnpy.lnpicollectionutils import build_grid


def test_build_grid0() -> None:
    x0 = -0.2124

    grid = build_grid(
        x=None,
        dx=0.5,
        x0=x0,
        x_range=(-2, 2),
        even_grid=True,
        digits=1,
        outlier=False,
    )

    assert grid == approx(np.arange(-2, +2 + 0.25, 0.5))


def test_build_grid1() -> None:
    x0 = -0.2124

    grid = build_grid(
        x=None,
        dx=0.5,
        x0=x0,
        offsets=(-2, +2),
        even_grid=True,
        digits=1,
        outlier=False,
    )

    assert grid == approx(np.arange(-2, +2, 0.5))
