"""
Typing definitions for :mod:`lnpy`
==================================
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from numpy.typing import ArrayLike, NDArray

    # Note: use fully qualified names
    import lnpy.lnpidata
    import lnpy.lnpiseries

    from .ensembles import xCanonical, xGrandCanonical


__all__ = [
    "T_Ensemble",
    "P",
    "R",
    "T",
    "FuncType",
    "F",
    "C_Ensemble",
    "MyNDArray",
    "IndexingInt",
    "xArrayLike",
    "T_Element",
    "T_SeriesWrapper",
    "TagPhasesSignature",
    "PhasesFactorySignature",
    "MaskConvention",
    "PeakStyle",
    "PeakError",
]


T_Ensemble = TypeVar("T_Ensemble", "xGrandCanonical", "xCanonical")
"""TypeVar for Ensemble."""

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
C_Ensemble: TypeAlias = Callable[Concatenate[T_Ensemble, P], R]

FuncType = Callable[..., Any]

F = TypeVar("F", bound=FuncType)


MyNDArray: TypeAlias = "NDArray[Any]"
"""Alias for simple :class:`numpy.typing.NDArray`"""


IndexingInt: TypeAlias = Union[
    int,
    "np.int_",
    "np.integer[Any]",
    "np.unsignedinteger[Any]",
    "np.signedinteger[Any]",
    "np.int8",
]


xArrayLike: TypeAlias = "ArrayLike | xr.DataArray"  # noqa: N816

IndexIterScalar: TypeAlias = Union[str, bytes, bool, int, float]
Scalar: TypeAlias = IndexIterScalar

# Series stuff
T_Element = TypeVar("T_Element", bound="lnpy.lnpidata.lnPiMasked")
"""TypeVar for element of lnpy.lnpiseries.SeriesWrapper."""

# T_SeriesWrapper = TypeVar("T_SeriesWrapper", bound="lnpy.lnpiseries.SeriesWrapper[T_Element]")
T_SeriesWrapper = TypeVar("T_SeriesWrapper", bound="lnpy.lnpiseries.SeriesWrapper")  # type: ignore[type-arg]
"""TypeVar for lnpy.lnpiseries.SeriesWrapper"""

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
