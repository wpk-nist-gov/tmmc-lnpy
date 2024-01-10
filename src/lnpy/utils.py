"""
Utility functions (:mod:`~lnpy.utils`)
======================================
"""

from __future__ import annotations

from functools import lru_cache
from itertools import starmap
from typing import TYPE_CHECKING, TypedDict, cast, overload

import numpy as np

from .docstrings import docfiller
from .options import OPTIONS

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from scipy.optimize import RootResults

    from lnpy.lnpidata import lnPiMasked

    from ._typing import MaskConvention, MyNDArray, R, T


# --- TQDM setup -----------------------------------------------------------------------
@lru_cache
def _get_tqdm() -> ModuleType | None:
    try:
        import tqdm

        return tqdm
    except ImportError:
        return None


@lru_cache
def _get_tqdm_default() -> Callable[..., Any]:
    _tqdm = _get_tqdm()
    if _tqdm:
        try:
            from IPython.core.getipython import (  # pyright: ignore[reportMissingImports]
                get_ipython,
            )

            p = get_ipython()  # type: ignore[no-untyped-call, unused-ignore]
            if p is not None and p.has_trait("kernel"):
                from tqdm.notebook import tqdm as tqdm_default

                return tqdm_default
            return cast("Callable[..., Any]", _tqdm.tqdm)
        except ImportError:
            return cast("Callable[..., Any]", _tqdm.tqdm)
    else:

        def wrapper(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:  # noqa: ARG001
            return seq

        return wrapper


def tqdm(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:
    opt = OPTIONS["tqdm_bar"]
    _tqdm = _get_tqdm()
    if _tqdm:
        if opt == "text":
            func = _tqdm.tqdm
        elif opt == "notebook":
            func = _tqdm.tqdm_notebook
        else:
            func = _get_tqdm_default()
    else:
        func = _get_tqdm_default()
    return cast("Iterable[T]", func(seq, *args, **kwargs))


def get_tqdm(
    seq: Iterable[T], len_min: str | int, leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    n = kwargs.get("total", None)
    _tqdm = _get_tqdm()

    if isinstance(len_min, str):
        len_min = OPTIONS[len_min]  # type: ignore[literal-required]

    if n is None:
        seq = tuple(seq)
        n = len(seq)

    if _tqdm and OPTIONS["tqdm_use"] and n >= len_min:
        if leave is None:
            leave = OPTIONS["tqdm_leave"]
        seq = tqdm(seq, leave=leave, **kwargs)
    return seq


def get_tqdm_calc(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_calc", leave=leave, **kwargs)


def get_tqdm_build(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_build", leave=leave, **kwargs)


# get_tqdm_calc = partial(get_tqdm, len_min="tqdm_len_calc")
# get_tqdm_build = partial(get_tqdm, len_min="tqdm_len_build")

# def get_tqdm_calc(seq, len_min=None, leave=None, **kwargs):
#     if len_min is None:
#         len_min = OPTIONS['tqdm_len_calc']
#     return _get_tqdm(seq, len_min=len_min, leave=leave, **kwargs)


# def get_tqdm_build(seq, len_min=None, leave=None, **kwargs):
#     if len_min is None:
#         len_min = OPTIONS['tqdm_len_build']
#     return _get_tqdm(seq, len_min=len_min, leave=leave, **kwargs)


# --------------------------------------------------
# JOBLIB stuff
@lru_cache
def _get_joblib() -> Any:
    try:
        import joblib
    except ImportError:
        joblib = None
    return joblib


def _use_joblib(
    items: Sequence[Any],
    len_key: str,
    use_joblib: bool = True,
    total: int | None = None,
) -> bool:
    joblib = _get_joblib()

    if use_joblib and joblib and OPTIONS["joblib_use"]:
        if total is None:
            total = len(items)

        return total >= OPTIONS[len_key]  # type: ignore[no-any-return,literal-required]
    return False


def _parallel(seq: Iterable[Any]) -> list[Any]:
    joblib = _get_joblib()

    return joblib.Parallel(  # type: ignore[no-any-return]
        n_jobs=OPTIONS["joblib_n_jobs"],
        backend=OPTIONS["joblib_backend"],
        **OPTIONS["joblib_kws"],
    )(seq)


def parallel_map_build(
    func: Callable[..., R], items: Iterable[Any], *args: Any, **kwargs: Any
) -> list[R]:
    joblib = _get_joblib()

    items = tuple(items)

    if _use_joblib(items, "joblib_len_build"):
        return _parallel(joblib.delayed(func)(x, *args, **kwargs) for x in items)
    return [func(x, *args, **kwargs) for x in items]


def _func_call(x: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    return x(*args, **kwargs)


def parallel_map_call(
    items: Sequence[Callable[..., Any]],
    use_joblib: bool,  # noqa: ARG001
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    joblib = _get_joblib()
    if _use_joblib(items, "joblib_len_calc"):
        return _parallel(joblib.delayed(_func_call)(x, *args, **kwargs) for x in items)
    return [x(*args, **kwargs) for x in items]


def parallel_map_attr(attr: str, use_joblib: bool, items: Sequence[Any]) -> list[Any]:  # noqa: ARG001
    from operator import attrgetter

    joblib = _get_joblib()
    func = attrgetter(attr)

    if _use_joblib(items, "joblib_len_calc"):
        return _parallel(joblib.delayed(func)(x) for x in items)
    return [func(x) for x in items]


def parallel_map_func_starargs(
    func: Callable[..., R],
    use_joblib: bool,  # noqa: ARG001
    items: Iterable[Any],
    total: int | None = None,
) -> list[R]:
    joblib = _get_joblib()

    items = tuple(items)

    if _use_joblib(items, "joblib_len_calc", total=total):
        return _parallel(starmap(joblib.delayed(func), items))
    return list(starmap(func, items))


# ----------------------------------------
# pandas stuff
def allbut(levels: Iterable[str], *names: str) -> list[str]:
    name_set = set(names)
    return [item for item in levels if item not in name_set]


# ----------------------------------------
# xarray utils
def dim_to_suffix_dataarray(
    da: xr.DataArray, dim: Hashable, join: str = "_"
) -> xr.Dataset:
    if dim in da.dims:
        return da.assign_coords(  # type: ignore[misc]
            **{dim: lambda x: [f"{x.name}{join}{c}" for c in x[dim].values]}  # type: ignore[arg-type]
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
) -> xr.DataArray:
    ...


@overload
def dim_to_suffix(ds: xr.Dataset, dim: Hashable = ..., join: str = ...) -> xr.Dataset:
    ...


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


def _convention_to_bool(convention: MaskConvention) -> bool:
    if convention == "image":
        convention = True
    elif convention == "masked":
        convention = False
    elif not isinstance(convention, bool):
        msg = f"Bad value {convention} sent to _convention_to_bool"
        raise ValueError(msg)
    return convention


@overload
def mask_change_convention(
    mask: None,
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> None:
    ...


@overload
def mask_change_convention(
    mask: MyNDArray,
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> MyNDArray:
    ...


@docfiller.decorate
def mask_change_convention(
    mask: MyNDArray | None,
    convention_in: MaskConvention = "image",
    convention_out: MaskConvention = "masked",
) -> MyNDArray | None:
    """
    Convert an array from one 'mask' convention to another.

    Parameters
    ----------
    {mask_general}
    convention_in, convention_out : string or bool
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * ``'image' ``or ``True`` : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * ``'masked'`` or ``False``:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

    Returns
    -------
    ndarray
        New 'mask' array with specified convention.
    """

    if mask is None:
        return mask

    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        mask = ~mask
    return mask


# m0: list[MyNDArray]
# reveal_type(masks_change_convention(m0))

# m1: list[None]
# reveal_type(masks_change_convention(m1))

# m2: list[MyNDArray | None]
# reveal_type(masks_change_convention(m2))


@overload
def masks_change_convention(
    masks: Sequence[MyNDArray],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> Sequence[MyNDArray]:
    ...


@overload
def masks_change_convention(
    masks: Sequence[None],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> Sequence[None]:
    ...


@overload
def masks_change_convention(
    masks: Sequence[MyNDArray | None],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> Sequence[MyNDArray | None]:
    ...


def masks_change_convention(
    masks: Sequence[MyNDArray] | Sequence[None] | Sequence[MyNDArray | None],
    convention_in: MaskConvention = "image",
    convention_out: MaskConvention = "masked",
) -> Sequence[MyNDArray] | Sequence[None] | Sequence[MyNDArray | None]:
    """
    Perform convention change of sequence of masks

    Parameters
    ----------
    masks : sequence of array-like
        masks[i] is the 'ith' mask
    convention_in, convention_out : string or bool or None
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * 'image' or True : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * 'masked' or False:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

        If `None`, then pass return input `mask`

    Returns
    -------
    new_masks : list of array
        New 'masks' array with specified convention.
    """

    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        masks = [m if m is None else ~m for m in masks]
    return masks


##################################################
# labels/masks utilities
##################################################
@docfiller.decorate
def labels_to_masks(
    labels: MyNDArray,
    features: Sequence[int] | MyNDArray | None = None,
    include_boundary: bool = False,
    convention: MaskConvention = "image",
    check_features: bool = True,
    **kwargs: Any,
) -> tuple[list[MyNDArray], Sequence[int]]:
    """
    Convert labels array to list of masks

    Parameters
    ----------
    labels : ndarray of int
        Each unique value `i` in `labels` indicates a mask.
        That is ``labels == i``.
    {features}
    {include_boundary}
    {mask_convention}
    {check_features}

    **kwargs
        arguments to find_boundary if include_boundary is True
        default to mode='outer', connectivity=labels.ndim

    Returns
    -------
    output : list of array of bool
        list of mask arrays, each with same shape as ``labels``, with
        mask convention `convention`.
    features : list
        features

    See Also
    --------
    masks_to_labels
    :func:`skimage.segmentation.find_boundaries`

    """
    from skimage import segmentation

    if include_boundary:
        kwargs = dict({"mode": "outer", "connectivity": labels.ndim}, **kwargs)
    if features is None:
        features = [i for i in np.unique(labels) if i > 0]

    elif check_features:
        vals = np.unique(labels)
        assert np.all([x in vals for x in features])

    convention = _convention_to_bool(convention)

    output: list[MyNDArray] = []
    for i in features:
        m: NDArray[np.bool_] = labels == i
        if include_boundary:
            # fmt: off
            b = cast(
                "NDArray[np.bool_]",
                segmentation.find_boundaries(m.astype(int), **kwargs),  # pyright: ignore[reportUnknownMemberType]
            )
            # fmt: on
            m = m | b
        if not convention:
            m = ~m
        output.append(m)
    return output, features  # type: ignore[return-value]


@docfiller.decorate
def masks_to_labels(
    masks: Sequence[MyNDArray],
    features: Sequence[int] | MyNDArray | None = None,
    convention: MaskConvention = "image",
    dtype: DTypeLike = np.int_,
) -> MyNDArray:
    """
    Convert list of masks to labels

    Parameters
    ----------
    masks : list of array-like of bool
        list of mask arrays.
    {features}
    {mask_convention}

    Returns
    -------
    ndarray
        Label array.


    See Also
    --------
    labels_to_masks
    """

    if features is None:
        features = range(1, len(masks) + 1)
    else:
        assert len(features) == len(masks)

    labels = np.full(masks[0].shape, fill_value=0, dtype=dtype)

    masks = masks_change_convention(masks, convention, True)

    for i, m in zip(features, masks):
        labels[m] = i
    return labels


def ffill(arr: MyNDArray, axis: int = -1, limit: int | None = None) -> MyNDArray:
    import bottleneck

    _limit = limit if limit is not None else arr.shape[axis]
    return bottleneck.push(arr, n=_limit, axis=axis)  # type: ignore[no-any-return]


def bfill(arr: MyNDArray, axis: int = -1, limit: int | None = None) -> MyNDArray:
    """Inverse of ffill"""
    import bottleneck

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    arr = np.flip(arr, axis=axis)
    # fill
    arr = bottleneck.push(arr, axis=axis, n=_limit)
    # reverse back to original
    return np.flip(arr, axis=axis)


##################################################
# calculations
##################################################


def get_lnz_iter(lnz: Iterable[float | None], x: ArrayLike) -> MyNDArray:
    """
    Create a lnz_iter object for varying a single lnz

    Parameters
    ----------
    lnz : list
        list with one element equal to None.  This is the component which will be varied
        For example, lnz=[lnz0,None,lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
        vary component 1

    x : array
        values to insert for variable component

    Returns
    -------
    ndarray
        Shape ``(len(x),len(lnz))``.
        array with rows [lnz0,lnz1,lnz2]
    """
    x = np.asarray(x)
    z = np.zeros_like(x)

    return np.array([x if m is None else z + m for m in lnz]).T


##################################################
# utilities
##################################################


# def sort_lnPis(values: Sequence[lnPiMasked], comp=0) -> list[lnPiMasked]:
#     """
#     Sort list of lnPi  that component `comp` mole fraction increases

#     Parameters
#     ----------
#     values : list of lnPiMasked

#     comp : int, default=0
#      component to sort along

#     Returns
#     -------
#     output : list of lnPiMasked
#         Objects in sorted order.
#     """

#     molfrac_comp = np.array([x.molfrac[comp] for x in values])

#     order = np.argsort(molfrac_comp)

#     output = [values[i] for i in order]

#     return output


def distance_matrix(
    mask: ArrayLike, convention: MaskConvention = "image"
) -> NDArray[np.float_]:
    """
    Create matrix of distances from elements of mask
    to nearest background point

    Parameters
    ----------
    mask : array-like
        image mask
    convention : str or bool, default='image'
        mask convention

    Returns
    -------
    distance : ndarray
        Same shape as mask.
        Distance from possible feature elements to background


    See Also
    --------
    ~scipy.ndimage.distance_transform_edt
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    mask = mask_change_convention(mask, convention_in=convention, convention_out=True)

    # pad mask
    # add padding to end of matrix in each dimension
    ndim = mask.ndim
    pad_width = ((0, 1),) * ndim
    mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=False)

    # distance filter
    dist = distance_transform_edt(mask)

    # remove padding
    s = (slice(None, -1),) * ndim
    return dist[s]  # type: ignore[no-any-return]


def lnpimasked_to_dataset(
    data: lnPiMasked, keys: Sequence[str] = ("lnpi", "PE")
) -> xr.Dataset:
    """
    Convert a :class:`~lnpy.lnpidata.lnPiMasked` object into as :class:`~xarray.Dataset`.

    Parameters
    ----------
    data : lnPiMasked

    Returns
    -------
    output : Dataset
    """

    return data.xce.table(keys=keys, default_keys=None)


def dataset_to_lnpimasked(
    ds: xr.Dataset,
    lnpi_name: str = "lnpi",
    pe_name: str = "PE",
    extra_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> lnPiMasked:
    """
    Convert a :class:`~xarray.Dataset` to a :class:`~lnpy.lnpidata.lnPiMasked` object.

    Parameters
    ----------
    ds : :class:`~xarray.Dataset`
    lnpi_name, pe_name : str
        Names of 'lnPi' and 'PE' parameters
    extra_kws : mapping, optional
        parameter `extra_kws`.  Note that if `pe_name` found in `ds`, then will add it to `extra_kws`.

    Returns
    -------
    lnpi : lnPiMasked
    """

    from .lnpidata import lnPiMasked

    data = ds[lnpi_name]

    if extra_kws is None:
        extra_kws = {}

    if pe_name in ds:
        extra_kws[pe_name] = ds[pe_name].values  # type: ignore[index]

    return lnPiMasked.from_dataarray(da=data, extra_kws=extra_kws, **kwargs)


# --- Root results ---------------------------------------------------------------------


class _RootResultDictReq(TypedDict, total=True):
    root: float
    iterations: int
    function_calls: int
    converged: bool
    flag: str


class RootResultDict(_RootResultDictReq, total=False):
    """Interface to :class:`scipy.optimize.RootResults`."""

    residual: float | MyNDArray


def rootresults_to_rootresultdict(
    r: RootResults, residual: float | MyNDArray | None
) -> RootResultDict:
    """Convert :class:`scipy.optimize.RootResults` to typed dictionary"""

    out = RootResultDict(
        root=r.root,
        iterations=r.iterations,
        function_calls=r.function_calls,
        converged=r.converged,
        flag=r.flag,
    )

    if residual is not None:
        out["residual"] = residual
    return out
