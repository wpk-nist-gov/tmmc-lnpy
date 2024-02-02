"""Version compatibility code should go here."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Hashable, Iterable

    from scipy.optimize import RootResults  # pyright: ignore[reportMissingTypeStubs]

    from ._typing import MyNDArray


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
    root: float | MyNDArray | None,
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
