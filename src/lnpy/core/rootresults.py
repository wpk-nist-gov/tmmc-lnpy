"""Results from root finding"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from scipy.optimize import RootResults

    from .typing import NDArrayAny


# --- Root results ---------------------------------------------------------------------


class RootResultDictReq(TypedDict, total=True):
    """Base root results"""

    root: float | NDArrayAny | None
    iterations: int
    function_calls: int
    converged: bool
    flag: str


class RootResultDict(RootResultDictReq, total=False):
    """Interface to :class:`scipy.optimize.RootResults`."""

    residual: float | NDArrayAny


def rootresults_to_rootresultdict(
    r: RootResults, residual: float | NDArrayAny | None
) -> RootResultDict:
    """Convert :class:`scipy.optimize.RootResults` to typed dictionary"""
    out = RootResultDict(
        root=r.root,
        iterations=r.iterations,
        function_calls=r.function_calls,
        converged=r.converged,
        flag=r.flag,  # pyright: ignore[reportArgumentType]
    )

    if residual is not None:
        out["residual"] = residual
    return out
