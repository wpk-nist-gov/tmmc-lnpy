"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lnpy.core.typing import FloatT, NDArrayInt

_PARALLEL = False
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

        out[index[start]] = 0.0
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

        accum = delta_lnpi[index[start]]
        lnpi[start] = accum

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

        accum = np.exp(lnpi[index[start]])
        for i in range(start + 1, end):
            s = index[i]
            accum += np.exp(lnpi[s])

        norm = np.log(accum)
        for i in range(start, end):
            s = index[i]
            out[s] = lnpi[s] - norm


@_decorator(
    "(n),(n),(window),(window) -> (window)",
    [
        (nb.float64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:]),
    ],
    writable=None,
)
def state_max_window(
    state: NDArray[FloatT],
    window_index: NDArrayInt,
    window_start: NDArrayInt,
    window_end: NDArrayInt,
    out: NDArray[FloatT],
) -> None:
    """Get the maximum state value for each window"""
    for window in range(len(window_start)):
        start = window_start[window]
        end = window_end[window]
        max_ = state[window_index[start]]
        for i in range(start + 1, end):
            max_ = max(max_, state[window_index[i]])
        out[window] = max_


@_decorator(
    "(n),(window),  (n),(window),(window),  (window),(rec),(rec) -> (n),()",
    [
        (
            nb.float64[:],  # state
            nb.float64[:],  # state_min
            nb.int64[:],  # window_index
            nb.int64[:],  # window_start
            nb.int64[:],  # window_end
            nb.int64[:],  # rec_index
            nb.int64[:],  # rec_start
            nb.int64[:],  # rec_end
            nb.int64[:],  # indexer
            nb.int64[:],  # count
        ),
    ],
    writable=None,
)
def keep_first_indexer(
    state: NDArray[FloatT],
    state_min: NDArray[FloatT],
    window_index: NDArrayInt,
    window_start: NDArrayInt,
    window_end: NDArrayInt,
    rec_index: NDArrayInt,
    rec_start: NDArrayInt,
    rec_end: NDArrayInt,
    indexer: NDArrayInt,
    count: NDArrayInt,
) -> None:
    """Get the maximum state value for each window"""
    m = 0
    indexer[:] = -1
    for rec in range(len(rec_start)):
        for i in range(rec_start[rec], rec_end[rec]):
            window = rec_index[i]
            min_ = state_min[window]

            for j in range(window_start[window], window_end[window]):
                s = window_index[j]
                if state[s] > min_:
                    indexer[m] = s
                    m += 1

    count[0] = m
