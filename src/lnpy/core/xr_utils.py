"""xarray utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from lnpy.core.validate import is_dataset

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from typing import Any

    import xarray as xr

    from .typing import (
        ApplyUFuncKwargs,
        AxisReduce,
        DimsReduce,
        MissingCoreDimOptions,
    )


# * apply_ufunc_kws
def factory_apply_ufunc_kwargs(
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    dask: str = "parallel",
    dask_gufunc_kwargs: Mapping[str, Any] | None = None,
    output_sizes: Mapping[Hashable, int] | None = None,
    output_dtypes: Any = None,
) -> dict[str, Any]:
    """
    Create kwargs to pass to :func:`xarray.apply_ufunc`

    Pass in options with ``apply_ufunc_kwargs``.  The other options set defaults of that parameter.
    """
    out: dict[str, Any] = {} if apply_ufunc_kwargs is None else dict(apply_ufunc_kwargs)

    out.setdefault("on_missing_core_dim", on_missing_core_dim)
    out.setdefault("dask", dask)
    out.setdefault(
        "dask_gufunc_kwargs",
        {} if dask_gufunc_kwargs is None else dict(dask_gufunc_kwargs),
    )

    if output_sizes:
        out["dask_gufunc_kwargs"].setdefault("output_sizes", dict(output_sizes))
    if output_dtypes:
        out.setdefault("output_dtypes", output_dtypes)
    return out


# * xarray utils --------------------------------------------------------------
def dim_to_suffix_dataarray(
    da: xr.DataArray, dim: Hashable, join: str = "_"
) -> xr.Dataset:
    if dim in da.dims:
        return da.assign_coords(  # type: ignore[misc]
            **{dim: lambda x: [f"{x.name}{join}{c}" for c in x[dim].to_numpy()]}  # type: ignore[arg-type]
        ).to_dataset(dim=dim)
    return da.to_dataset()


def dim_to_suffix_dataset(
    table: xr.Dataset, dim: Hashable, join: str = "_"
) -> xr.Dataset:
    out = table
    for k in out:
        if dim in out[k].dims:
            out = out.drop_vars(k).update(  # type: ignore[arg-type,unused-ignore]
                table[k].pipe(dim_to_suffix_dataarray, dim, join)
            )
    return out


@overload
def dim_to_suffix(
    ds: xr.DataArray, dim: Hashable = ..., join: str = ...
) -> xr.DataArray: ...


@overload
def dim_to_suffix(
    ds: xr.Dataset, dim: Hashable = ..., join: str = ...
) -> xr.Dataset: ...


def dim_to_suffix(
    ds: xr.DataArray | xr.Dataset, dim: Hashable = "component", join: str = "_"
) -> xr.DataArray | xr.Dataset:
    from xarray import DataArray, Dataset

    if isinstance(ds, DataArray):
        return dim_to_suffix_dataarray(ds, dim=dim, join=join)
    if isinstance(ds, Dataset):
        return dim_to_suffix_dataset(ds, dim=dim, join=join)
    msg = "`ds` must be `DataArray` or `Dataset`"
    raise ValueError(msg)


# * Select Axis ---------------------------------------------------------------
def select_axis_dim(
    target: xr.DataArray | xr.Dataset, axis: AxisReduce, dim: DimsReduce | None
) -> tuple[int, DimsReduce]:
    if is_dataset(target):
        if dim is None:
            msg = "Must specify `dim` with dataset"
            raise ValueError(msg)
        return -1, dim

    if dim is not None:
        axis = target.get_axis_num(dim)  # type: ignore[assignment]
    else:
        dim = target.dims[axis]
    return axis, dim
