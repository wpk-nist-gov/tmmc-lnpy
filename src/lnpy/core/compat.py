"""Version compatibility code should go here."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from typing import Any

    from scipy.optimize import RootResults  # pyright: ignore[reportMissingTypeStubs]

    from .typing import NDArrayAny


if sys.version_info >= (3, 10):
    from importlib import resources
else:
    import importlib_resources as resources


_COPY_IF_NEEDED = None if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else False


def copy_if_needed(
    copy: bool | None,
) -> bool:  # Lie here so can support both versions...
    """Callable to return copy if needed convention..."""
    if not copy:
        return _COPY_IF_NEEDED  # type: ignore[return-value]
    return copy


__all__ = ["resources", "rootresults", "xr_dot"]


def xr_dot(
    *arrays: Any,
    dim: str | Iterable[Hashable] | "ellipsis" | None = None,  # noqa: F821, UP037
    **kwargs: Any,
) -> Any:
    """
    Interface to xarray.dot.

    Remove a deprecation warning for older xarray versions.
    xarray deprecated `dims` keyword.  Use `dim` instead.
    """
    import xarray as xr

    try:
        return xr.dot(*arrays, dim=dim, **kwargs)  # type: ignore[arg-type,unused-ignore]
    except TypeError:
        return xr.dot(*arrays, dims=dim, **kwargs)  # type: ignore[arg-type,unused-ignore]


def rootresults(
    root: float | NDArrayAny | None,
    iterations: int,
    function_calls: int,
    flag: int,
    method: str = "User",
) -> RootResults:
    """
    Create :class:`scipy.optimize.RootResults` object.

    There are differences in `RootResults` for different
    versions of scipy, so have this as an interface
    """
    from scipy.optimize import RootResults  # pyright: ignore[reportMissingTypeStubs]

    try:
        return RootResults(
            root=root,
            iterations=iterations,
            function_calls=function_calls,
            flag=flag,
            method=method,  # pyright: ignore[reportCallIssue]
        )
    except TypeError:
        return RootResults(  # pyright: ignore[reportCallIssue]
            root=root,
            iterations=iterations,
            function_calls=function_calls,
            flag=flag,
        )
