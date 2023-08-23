from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from numpy.typing import ArrayLike, NDArray

    from .ensembles import xCanonical, xGrandCanonical
    from .lnpidata import lnPiMasked
    from .lnpiseries import SeriesWrapper, lnPiCollection


T_Ensemble = TypeVar("T_Ensemble", "xGrandCanonical", "xCanonical")
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
C_Ensemble: TypeAlias = Callable[Concatenate[T_Ensemble, P], R]

FuncType = Callable[..., Any]

F = TypeVar("F", bound=FuncType)


T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", "xr.DataArray", "xr.Dataset")


MyNDArray: TypeAlias = "NDArray[Any]"


IndexingInt: TypeAlias = Union[
    int,
    "np.int_",
    "np.integer[Any]",
    "np.unsignedinteger[Any]",
    "np.signedinteger[Any]",
    "np.int8",
]


xArrayLike: TypeAlias = "ArrayLike | xr.DataArray"

IndexIterScalar: TypeAlias = Union[str, bytes, bool, int, float]
Scalar: TypeAlias = IndexIterScalar

# Series stuff
T_Element = TypeVar("T_Element", bound="lnPiMasked")
T_SeriesWrapper = TypeVar("T_SeriesWrapper", bound="SeriesWrapper[T_Element]")  # type: ignore


# Segmentation stuff
PeakStyle = Literal["indices", "mask", "marker"]
PeakError = Literal["ignore", "raise", "warn"]


TagPhasesSignature = Callable[[Sequence["lnPiMasked"]], "Sequence[int] | MyNDArray"]
PhasesFactorySignature = Callable[..., "lnPiCollection"]


# Mask convention
MaskConvention = Literal["image", "masked", True, False]
