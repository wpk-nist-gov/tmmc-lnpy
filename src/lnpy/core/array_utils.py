"""Array utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from .validate import is_dataset, is_ndarray

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike

    from .typing import NDArrayAny


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


@overload
def select_dtype(
    x: xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...
@overload
def select_dtype(
    x: xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64]: ...
@overload
def select_dtype(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...


def select_dtype(
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None:  # DTypeLikeArg[Any]:
    """
    Select a dtype from, in order, out, dtype, or passed array.

    If pass in a Dataset, return dtype
    """
    if is_dataset(x):
        if dtype is None:
            return dtype
        dtype = np.dtype(dtype)
    elif out is not None:
        dtype = out.dtype  # pyright: ignore[reportUnknownMemberType]
    elif dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = getattr(x, "dtype", np.dtype(np.float64))

    if dtype in _ALLOWED_FLOAT_DTYPES:
        return dtype  # type: ignore[return-value]

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def asarray_maybe_recast(
    data: ArrayLike, dtype: DTypeLike = None, recast: bool = False
) -> NDArrayAny:
    """Perform asarray with optional recast to `dtype` if not already an array."""
    if is_ndarray(data):
        if recast and dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data
    return np.asarray(data, dtype=dtype)
