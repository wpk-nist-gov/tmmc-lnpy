"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lnpy._typing import FloatT, NDArrayInt


_PARALLEL = True  # Auto generated from combine.py
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(index), (index), (index), (group), (group) -> (index)",
    [
        (
            nb.float32[:],
            nb.float32[:],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float32[:],
        ),
        (
            nb.float64[:],
            nb.float64[:],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float64[:],
        ),
    ],
    writable=None,
)
def delta_lnpi_from_updown(
    down: NDArray[FloatT],
    up: NDArray[FloatT],
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    out: NDArray[FloatT],
) -> None:
    ngroups = len(group_start)

    for group in range(ngroups):
        start = group_start[group]
        end = group_end[group]

        out[start] = 0.0
        if end > start:
            up_last = up[start]

            for i in range(start + 1, end):
                s = index[i]
                out[s] = np.log(up_last / down[s])
                up_last = up[s]


@_decorator(
    "(index), (index), (group), (group) -> (index)",
    [
        (nb.float32[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float32[:]),
        (nb.float64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:]),
    ],
    writable=None,
)
def lnpi_from_delta_lnpi(
    delta_lnpi: NDArray[FloatT],
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    lnpi: NDArray[FloatT],
) -> None:
    ngroups = len(group_start)

    for group in range(ngroups):
        start = group_start[group]
        end = group_end[group]

        accum = delta_lnpi[start]
        lnpi[start] = accum

        if end > start:
            for i in range(start + 1, end):
                s = index[i]
                accum += delta_lnpi[i]
                lnpi[s] = accum


@_decorator(
    "(index), (index), (group), (group) -> (index)",
    [
        (nb.float32[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float32[:]),
        (nb.float64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:]),
    ],
    writable=None,
)
def normalize_lnpi(
    lnpi: NDArray[FloatT],
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    out: NDArray[FloatT],
) -> None:
    ngroups = len(group_start)

    for group in range(ngroups):
        start = group_start[group]
        end = group_end[group]

        accum = np.exp(lnpi[start])
        if end > start:
            for i in range(start + 1, end):
                s = index[i]
                accum += np.exp(lnpi[s])

        norm = np.log(accum)
        for i in range(start, end):
            s = index[i]
            out[s] = lnpi[s] - norm
