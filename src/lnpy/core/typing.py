"""
Typing definitions for :mod:`lnpy`
==================================
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .typing_compat import Concatenate, ParamSpec, TypeAlias, TypeVar

if TYPE_CHECKING:
    # Note: use fully qualified names
    import lnpy.lnpidata
    import lnpy.lnpiseries
    from lnpy.ensembles import CanonicalEnsemble, GrandCanonicalEnsemble  # noqa: F401


__all__ = [
    "C_Ensemble",
    "EnsembleT",
    "F",
    "FuncType",
    "IndexingInt",
    "MaskConvention",
    "NDArrayAny",
    "P",
    "PeakError",
    "PeakStyle",
    "PhasesFactorySignature",
    "R",
    "T",
    "TagPhasesSignature",
    "XArrayLike",
]


FloatT = TypeVar("FloatT", np.float32, np.float64, default=Any)  # type: ignore[misc]
NDArrayInt = NDArray[np.int64]


EnsembleT = TypeVar("EnsembleT", "GrandCanonicalEnsemble", "CanonicalEnsemble")
"""TypeVar for Ensemble."""

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
C_Ensemble: TypeAlias = Callable[Concatenate[EnsembleT, P], R]

FuncType = Callable[..., Any]

F = TypeVar("F", bound=FuncType)


IndexingInt: TypeAlias = Union[
    int,
    "np.int_",
    "np.integer[Any]",
    "np.unsignedinteger[Any]",
    "np.signedinteger[Any]",
    "np.int8",
]


XArrayLike: TypeAlias = Union[ArrayLike, xr.DataArray]

IndexIterScalar: TypeAlias = Union[str, bytes, bool, int, float]
Scalar: TypeAlias = IndexIterScalar

# Segmentation stuff
PeakStyle = Literal["indices", "mask", "marker"]
PeakError = Literal["ignore", "raise", "warn"]

# reduction dimensions/axes
AxisReduce: TypeAlias = int
DimsReduce: TypeAlias = Union[Hashable, Collection[Hashable]]

NDArrayAny: TypeAlias = NDArray[Any]
"""Alias for simple :class:`numpy.typing.NDArray[Any]`"""

NDArrayBool: TypeAlias = NDArray[np.bool_]

TagPhasesSignature = Callable[
    [Sequence["lnpy.lnpidata.lnPiMasked"]], Union[Sequence[int], NDArrayAny]
]
"""Signature for tag_phases function."""

PhasesFactorySignature = Callable[..., "lnpy.lnpiseries.lnPiCollection"]
"""Signature for phases_factory function."""


MaskConvention = Literal["image", "masked", True, False]
"""Convention for boolean masks."""


# new stuff:
Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]

ApplyUFuncKwargs: TypeAlias = Mapping[str, Any]

MissingCoreDimOptions = Literal["raise", "copy", "drop"]

KeepAttrs: TypeAlias = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    bool,
    None,
]
