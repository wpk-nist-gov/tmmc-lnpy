"""
Typing definitions for :mod:`lnpy`
==================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar, Union

import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ._typing_compat import Concatenate, ParamSpec, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    # Note: use fully qualified names
    import lnpy.lnpidata
    import lnpy.lnpiseries

    from .ensembles import xCanonical, xGrandCanonical


__all__ = [
    "C_Ensemble",
    "F",
    "FuncType",
    "IndexingInt",
    "MaskConvention",
    "MyNDArray",
    "P",
    "PeakError",
    "PeakStyle",
    "PhasesFactorySignature",
    "R",
    "T",
    "T_Ensemble",
    "TagPhasesSignature",
    "xArrayLike",
]


T_Ensemble = TypeVar("T_Ensemble", "xGrandCanonical", "xCanonical")
"""TypeVar for Ensemble."""

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
C_Ensemble: TypeAlias = Callable[Concatenate[T_Ensemble, P], R]

FuncType = Callable[..., Any]

F = TypeVar("F", bound=FuncType)


MyNDArray: TypeAlias = NDArray[Any]
"""Alias for simple :class:`numpy.typing.NDArray`"""


IndexingInt: TypeAlias = Union[
    int,
    "np.int_",
    "np.integer[Any]",
    "np.unsignedinteger[Any]",
    "np.signedinteger[Any]",
    "np.int8",
]


xArrayLike: TypeAlias = Union[ArrayLike, xr.DataArray]  # noqa: N816

IndexIterScalar: TypeAlias = Union[str, bytes, bool, int, float]
Scalar: TypeAlias = IndexIterScalar

# Segmentation stuff
PeakStyle = Literal["indices", "mask", "marker"]
PeakError = Literal["ignore", "raise", "warn"]


TagPhasesSignature = Callable[
    [Sequence["lnpy.lnpidata.lnPiMasked"]], Union[Sequence[int], MyNDArray]
]
"""Signature for tag_phases function."""

PhasesFactorySignature = Callable[..., "lnpy.lnpiseries.lnPiCollection"]
"""Signature for phases_factory function."""


MaskConvention = Literal["image", "masked", True, False]
"""Convention for boolean masks."""
