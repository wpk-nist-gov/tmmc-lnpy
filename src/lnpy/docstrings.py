"""Common docstrings."""

from module_utilities.docfiller import DocFiller

_docstrings = """
Parameters
----------
copy : bool, optional
    If True, copy the data.  If False, attempt to use view.
copy_kws : mapping, optional
    extra arguments to copy
lnz : float or sequence of float
    Value(s) of ``lnz`` (log of activity)
state_kws : Mapping, optiona
    'State' variables.  Common key/value pairs used by other methods:

    * beta (float) : inverse temperature
    * volume (float) : volume
extra_kws : Mapping, Optional
    Extra parameters.  These are passed along to new objects without modification.
    Common key/value pairs:

* PE (array-like) : value of total potential energy.
    Used in calculation of canonical and grand canonical average potential energy.
fill_value : scalar, default=nan
    Value to fill for masked elements.
data : array-like
    Value of lnPi.  Number of dimensions should be same as ``len(lnz)``.
    For single component, ``data[k]`` is the value of lnPi for `k` particles.
mask_masked | mask : bool or array-like of bool, optional
    Where `mask` is `True`, values are "masked out".
mask_image | mask : bool or array-like of bool, optional
    This uses the "image" convention.  Where `mask` is `True`, values are include.
masks_masked | masks : sequence of array-like of bool
    Value of `masks[i]` is the mask for ith lnPi.
    masks[i] is True where values are "masked out".
mask_convention | convention : string or bool
    Convention for mask.  Allowable values are:

    * 'image' or True : `True` values included, `False` values excluded.
        This is the normal convention in :mod:`scipy.ndimage`.
    * 'masked' or False:  `False` values are included, `True` values are excluded.
        This is the convention in :mod:`numpy.ma`
labels : array-like of int
    Where `labels == k`, indicates regions where data corresponds to the kth label.
base : object
    lnPi data object.
peak_kws : mapping, optional
    Optional parameters to :func:`~lnpy.segment.peak_local_max_adaptive`
watershed_kws : mapping, optional
    Optional parameters to :func:`~skimage.segmentation.watershed`
"""


docfiller = DocFiller.from_docstring(_docstrings, combine_keys="parameters")
