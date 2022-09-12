"""
utility functions
"""

from functools import partial

import numpy as np

# import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from skimage import segmentation

# --------------------------------------------------
# TQDM stuff
try:
    import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

if _HAS_TQDM:
    try:
        from IPython import get_ipython

        p = get_ipython()
        if p is not None and p.has_trait("kernel"):
            from tqdm.notebook import tqdm as tqdm_default
        else:
            tqdm_default = _tqdm.tqdm
    except ImportError:
        tqdm_default = _tqdm.tqdm

from .options import OPTIONS


def tqdm(*args, **kwargs):
    opt = OPTIONS["tqdm_bar"]
    if opt == "text":
        func = _tqdm.tqdm
    elif opt == "notebook":
        func = _tqdm.tqdm_notebook
    else:
        func = tqdm_default

    return func(*args, **kwargs)


def get_tqdm(seq, len_min, leave=None, **kwargs):

    n = kwargs.get("total", None)

    if isinstance(len_min, str):
        len_min = OPTIONS[len_min]

    if n is None:
        n = len(seq)
    if _HAS_TQDM and OPTIONS["tqdm_use"] and n >= len_min:
        if leave is None:
            leave = OPTIONS["tqdm_leave"]
        seq = tqdm(seq, leave=leave, **kwargs)
    return seq


get_tqdm_calc = partial(get_tqdm, len_min="tqdm_len_calc")
get_tqdm_build = partial(get_tqdm, len_min="tqdm_len_build")

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
try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


from operator import attrgetter


def parallel_map_build(func, items, *args, **kwargs):
    if (
        _HAS_JOBLIB
        and OPTIONS["joblib_use"]
        and len(items) >= OPTIONS["joblib_len_build"]
    ):
        return Parallel(
            n_jobs=OPTIONS["joblib_n_jobs"],
            backend=OPTIONS["joblib_backend"],
            **OPTIONS["joblib_kws"]
        )(delayed(func)(x, *args, **kwargs) for x in items)
    else:
        return [func(x, *args, **kwargs) for x in items]


def _func_call(x, *args, **kwargs):
    return x(*args, **kwargs)


def parallel_map_call(items, use_joblib, *args, **kwargs):
    if (
        use_joblib
        and _HAS_JOBLIB
        and OPTIONS["joblib_use"]
        and len(items) >= OPTIONS["joblib_len_calc"]
    ):
        return Parallel(
            n_jobs=OPTIONS["joblib_n_jobs"],
            backend=OPTIONS["joblib_backend"],
            **OPTIONS["joblib_kws"]
        )(delayed(_func_call)(x, *args, **kwargs) for x in items)
    else:
        return [x(*args, **kwargs) for x in items]


def parallel_map_attr(attr, use_joblib, items):
    func = attrgetter(attr)
    if (
        use_joblib
        and _HAS_JOBLIB
        and OPTIONS["joblib_use"]
        and len(items) >= OPTIONS["joblib_len_calc"]
    ):
        return Parallel(
            n_jobs=OPTIONS["joblib_n_jobs"],
            backend=OPTIONS["joblib_backend"],
            **OPTIONS["joblib_kws"]
        )(delayed(func)(x) for x in items)
    else:
        return [func(x) for x in items]


def parallel_map_func_starargs(func, use_joblib, items, total=None):

    if total is None:
        total = len(items)

    if (
        use_joblib
        and _HAS_JOBLIB
        and OPTIONS["joblib_use"]
        and total >= OPTIONS["joblib_len_calc"]
    ):
        return Parallel(
            n_jobs=OPTIONS["joblib_n_jobs"],
            backend=OPTIONS["joblib_backend"],
            **OPTIONS["joblib_kws"]
        )(delayed(func)(*x) for x in items)
    else:
        return [func(*x) for x in items]


# ----------------------------------------
# pandas stuff
def allbut(levels, *names):
    names = set(names)
    return [item for item in levels if item not in names]


# ----------------------------------------
# xarray utils
def dim_to_suffix_dataarray(da, dim, join="_"):
    if dim in da.dims:
        return da.assign_coords(
            **{dim: lambda x: ["{}{}{}".format(x.name, join, c) for c in x[dim].values]}
        ).to_dataset(dim=dim)
    else:
        return da.to_dataset()


def dim_to_suffix_dataset(table, dim, join="_"):
    out = table
    for k in out:
        if dim in out[k].dims:
            out = out.drop(k).update(table[k].pipe(dim_to_suffix_dataarray, dim, join))
    return out


def dim_to_suffix(ds, dim="component", join="_"):
    if isinstance(ds, xr.DataArray):
        f = dim_to_suffix_dataarray
    elif isinstance(ds, xr.Dataset):
        f = dim_to_suffix_dataset
    else:
        raise ValueError("ds must be `DataArray` or `Dataset`")
    return f(ds, dim=dim, join=join)


def _convention_to_bool(convention):
    if convention == "image":
        convention = True
    elif convention == "masked":
        convention = False
    else:
        assert convention in [True, False]
    return convention


def mask_change_convention(mask, convention_in="image", convention_out="masked"):
    """
    convert an array from one 'mask' convention to another.

    Parameters
    ----------
    mask : array-like
        Masking array.
    convention_in, convention_out : string or bool or None.
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * 'image' or True : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * 'masked' or False:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

        If `None`, then pass return input `mask`

    Returns
    -------
    new_mask : array
    New 'mask' array with specified convension.
    """

    if mask is None:
        return mask

    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        mask = ~mask
    return mask


def masks_change_convention(masks, convention_in="image", convention_out="masked"):
    """
    Perform convension change of sequence of masks

    Parameters
    ----------
    masks : sequence of array-like
        masks[i] is the 'ith' mask
    convention_in, convention_out : string or bool or None.
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * 'image' or True : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * 'masked' or False:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

        If `None`, then pass return input `mask`

    Returns
    -------
    new_masks : list of arrays
        New 'masks' array with specified convension.
    """

    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        masks = [m if m is None else ~m for m in masks]
    return masks


##################################################
# labels/masks utilities
##################################################
def labels_to_masks(
    labels,
    features=None,
    include_boundary=False,
    convention="image",
    check_features=True,
    **kwargs
):
    """
    convert labels array to list of masks

    Parameters
    ----------
    labels : array of labels to analyze
        Each unique value `i` in `labels` indicates a mask.
        That is ``labels == i``.
    features : array-like, optional
        list of features to extract from labels.  Note that returned
        mask[i] corresponds to labels == feature[i].
    include_boundary : bool, default=False
        if True, include boundary regions in output mask
    convention : {'image','masked'} or bool.
        convention for output masks
    check_features : bool, default=True
        if True, and supply features, then make sure each feature is in labels

    **kwargs
        arguments to find_boundary if include_boundary is True
        default to mode='outer', connectivity=labels.ndim

    Returns
    -------
    output : list of masks of same shape as labels
        mask for each feature
    features : list
        features

    See Also
    --------
    masks_to_labels
    :func:`skimage.segmentation.find_boundaries`

    """

    if include_boundary:
        kwargs = dict(dict(mode="outer", connectivity=labels.ndim), **kwargs)
    if features is None:
        features = [i for i in np.unique(labels) if i > 0]
    elif check_features:
        vals = np.unique(labels)
        assert np.all([x in vals for x in features])

    convention = _convention_to_bool(convention)

    output = []
    for i in features:
        m = labels == i
        if include_boundary:
            b = segmentation.find_boundaries(m.astype(int), **kwargs)
            m = m + b
        if not convention:
            m = ~m
        output.append(m)
    return output, features


def masks_to_labels(masks, features=None, convention="image", dtype=int, **kwargs):
    """
    convert list of masks to labels

    Parameters
    ----------
    masks : list-like of masks

    features : value for each feature, optional
        labels[mask[i]] = features[i] + feature_offset
        Default = range(1, len(masks) + 1)

    convention : {'image','masked'} or bool
        convention of masks

    Returns
    -------
    labels : array of labels


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


def ffill(arr, axis=-1, limit=None):
    import bottleneck

    _limit = limit if limit is not None else arr.shape[axis]
    return bottleneck.push(arr, n=_limit, axis=axis)


def bfill(arr, axis=-1, limit=None):
    """inverse of ffill"""
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


def get_lnz_iter(lnz, x):
    """
    create a lnz_iter object for varying a single lnz

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
    ouptut : array of shape (len(x),len(lnz))
       array with rows [lnz0,lnz1,lnz2]
    """

    z = np.zeros_like(x)

    x = np.asarray(x)

    L = []
    for m in lnz:
        if m is None:
            L.append(x)
        else:
            L.append(z + m)

    return np.array(L).T


##################################################
# utilities
##################################################


def sort_lnPis(input, comp=0):
    """
    sort list of lnPi  that component `comp` mol fraction increases

    Parameters
    ----------
    input : list of lnPi objects


    comp : int (Default 0)
     component to sort along

    Returns
    -------
    output : list of lnPi objects in sorted order
    """

    molfrac_comp = np.array([x.molfrac[comp] for x in input])

    order = np.argsort(molfrac_comp)

    output = [input[i] for i in order]

    return output


def distance_matrix(mask, convention="image"):
    """
    create matrix of distances from elements of mask
    to nearest background point

    Parameters
    ----------
    mask : array-like
        image mask
    convention : str or bool, default='image'
        mask convetion

    Returns
    -------
    distance : array of same shape as mask
        distance from possible feature elements to background


    See Also
    --------
    scipy.ndimage.distance_transform_edt
    """

    mask = np.asarray(mask, dtype=bool)
    mask = masks_change_convention(mask, convention_in=convention, convention_out=True)

    # pad mask
    # add padding to end of matrix in each dimension
    ndim = mask.ndim
    pad_width = ((0, 1),) * ndim
    mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=False)

    # distance filter
    dist = ndi.distance_transform_edt(mask)

    # remove padding
    s = (slice(None, -1),) * ndim
    return dist[s]


def lnpimasked_to_dataset(data, keys=("lnpi", "PE")):
    """
    Convert a :class:`~lnpy.lnPiMasked` object into as :class:`~xarray.Dataset`.

    Parameters
    ----------
    data : lnPiMasked

    Returns
    -------
    output : Dataset
    """

    return data.xce.table(keys=keys, default_keys=None)


def dataset_to_lnpimasked(ds, lnpi_name="lnpi", pe_name="PE", extra_kws=None, **kwargs):
    """
    Convert a :class:`~xarray.Dataset` to a :class:`~lnpy.lnPiMasked` object.

    Parameters
    ----------
    ds : Dataset

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
        extra_kws[pe_name] = ds[pe_name].values

    return lnPiMasked.from_dataarray(da=data, extra_kws=extra_kws, **kwargs)
