from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray


T = TypeVar("T")

FuncType = Callable[..., Any]

F = TypeVar("F", bound=FuncType)


T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", "xr.DataArray", "xr.Dataset")


MyNDArray: TypeAlias = "NDArray[Any]"
