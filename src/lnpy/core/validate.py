"""Validations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from .typing_compat import TypeIs


# * TypeGuards ----------------------------------------------------------------
def is_ndarray(x: Any) -> TypeIs[NDArray[Any]]:
    """Typeguard ndarray."""
    return isinstance(x, np.ndarray)


def is_dataarray(x: object) -> TypeIs[xr.DataArray]:
    """Typeguard dataarray."""
    return isinstance(x, xr.DataArray)


def is_dataset(x: object) -> TypeIs[xr.Dataset]:
    """Typeguard dataset"""
    return isinstance(x, xr.Dataset)


def is_xarray(x: object) -> TypeIs[xr.Dataset | xr.DataArray]:
    """Typeguard xarray object"""
    return isinstance(x, (xr.DataArray, xr.Dataset))


# def is_xarray_typevar(x: ArrayLike | DataT) -> TypeIs[DataT]:
#     """Typeguard ``DataT`` typevar against array-like."""
#     return isinstance(x, (xr.DataArray, xr.Dataset))  # noqa: ERA001


def is_series(x: object) -> TypeIs[pd.Series[Any]]:
    """Typeguard pd.Series."""
    return isinstance(x, pd.Series)


def is_dataframe(x: object) -> TypeIs[pd.DataFrame]:
    """Typeguard pd.DataFrame."""
    return isinstance(x, pd.DataFrame)
