"""Common docstrings."""

from textwrap import dedent

import pandas.util._decorators as pd_dec

from .options import OPTIONS


def docfiller(*args, **kwargs):
    """To fill common docs.

    Taken from pandas.utils._decorators

    """

    if OPTIONS["doc_sub"]:
        return pd_dec.doc(*args, **kwargs)
    else:

        def decorated(func):
            return func

        return decorated


# fmt: off
_shared_docs = {
    "copy":
    """
    copy : bool, optional
        If True, copy the data.  If False, attempt to use view.
    """,
    "copy_kws":
    """
    copy_kws : mapping, optional
        extra arguments to copy
    """,
    "lnz":
    """
    lnz: float or sequence of floats
        Value(s) of ``lnz`` (log of activity)
    """,
    "state_kws":
    """
    state_kws : Mapping, optiona
        'State' variables.  Common key/value pairs used by other methods:

        * beta (float) : inverse temperature
        * volume (float) : volume
    """,
    "extra_kws":
    """
    extra_kws : Mapping, Optional
        Extra parameters.  These are passed along to new objects without modification.
        Common key/value pairs:

    * PE (array-like) : value of total potential energy.
      Used in calculation of canonical and grand canonical average potential energy.
    """,
    "fill_value":
    """
    fill_value : scalar, default=nan
        Value to fill for masked elements.
    """,
    "data":
    """
    data : array-like
        Value of lnPi.  Number of dimensions should be same as ``len(lnz)``.
        For single component, ``data[k]`` is the value of lnPi for `k` particles.
    """,
    "mask_masked":
    """
    mask : bool or array-like, optional
        Where `mask` is `True`, values are "masked out".
    """,
    "mask_image":
    """
    mask : bool or array-like, optional
        This uses the "image" convention.  Where `mask` is `True`, values are include.
    """,
    "masks_masked":
    """
    masks : sequence of array-like of bools
        Value of `masks[i]` is the mask for ith lnPi.
        masks[i] is True where values are "masked out".
    """,
    "mask_convention":
    """
    convention : string or bool
        Convention for mask.  Allowable values are:

        * 'image' or True : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * 'masked' or False:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`
    """,
    "labels":
    """
    labels : array-like of ints
        Where `labels == k`, indicates regions where data corresponds to the kth label.
    """,
    "base":
    """
    base : Data
        lnPi data object.
    """,
    "peak_kws":
    """
    peak_kws : mapping, optional
        Optional parameters to :func:`~lnpy.segment.peak_local_max_adaptive`
    """,

    "watershed_kws":
    """
    watershed_kws : mapping, optional
        Optional parameters to :func:`~skimage.segmentation.watershed`
    """,




}
# fmt: on


def _prepare_shared_docs(shared_docs):
    return {k: dedent(v).strip() for k, v in shared_docs.items()}


_shared_docs = _prepare_shared_docs(_shared_docs)


# add in xr_params
# _shared_docs["xr_params"] = "\n".join(
#     [
#         _shared_docs[k]
#         for k in ["dims", "mom_dims", "attrs", "coords", "name", "indexes", "template"]
#     ]
# )


docfiller_shared = docfiller(**_shared_docs)
