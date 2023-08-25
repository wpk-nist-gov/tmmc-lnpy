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
mask_masked | mask : None or ndarray of bool
    Mask using "masked" convention. Where `mask` is `True`, values are excluded.
mask_image | mask : None or ndarray of bool
    Mask using "image" convention. Where `mask` is `True`, values are included.
masks_masked | masks : sequence of None or ndarray of bool
    Masks using "masked" convention.  Where `mask[i]` is `True`, values are excluded
    for sample `i`.
masks_image | masks : sequence of None or ndarray of bool
    Masks using "image" convention.  Where `masks[i]` is `True`, values are included
    for sample `i`.
masks_general | masks: sequence of None or ndarray of bool
    Masking arrays.
mask_general | masks: None or ndarray of bool
    Masking array.
mask_convention | convention : string or bool
    Convention for mask.  Allowable values are:

    * 'image' or True : `True` values included, `False` values excluded.
        This is the normal convention in :mod:`scipy.ndimage`.
    * 'masked' or False:  `False` values are included, `True` values are excluded.
        This is the convention in :mod:`numpy.ma`
labels : np.ndarray of int
    Where `labels == k`, indicates regions where data corresponds to the kth label.
base : object
    lnPi data object.
peak_kws : mapping, optional
    Optional parameters to :func:`~lnpy.segment.peak_local_max_adaptive`
watershed_kws : mapping, optional
    Optional parameters to :func:`~skimage.segmentation.watershed`
features : sequence of int
    If specified, extract only those locations where ``labels == feature``
    for all values ``feature in features``.  That is, select a subset of
    unique label values.
include_boundary : bool
    if True, include boundary regions in output mask
check_features : bool
    if True, then make sure each feature is in labels
labels : ndarray of int
    Each unique value `i` in `labels` indicates a mask.
    That is ``labels == i`` is a mask for feature `i`.
ffill : bool, default=True
    Do forward filling
bfill : bool, default=False
    Do back filling
fill_limit | limit : int, default None
    The maximum number of consecutive NaN values to forward fill. In
    other words, if there is a gap with more than this number of
    consecutive NaNs, it will only be partially filled. Must be greater
    than 0 or None for no limit.
find_boundary_connectivity  | connectivity: int, optional
    Defaults to ``masks[0].ndim``
energy_idx | idx : int
    phase index to consider transitions from
energy_idx_nebr | idx_nebr : int or list, optional
    if supplied, consider transition from idx to idx_nebr or minimum of all element in idx_nebr.
    Default behavior is to return minimum transition from idx to all other neighboring regions
"""


_accessor_docs = {
    "xge": "Accessor to :class:`~lnpy.ensembles.xGrandCanonical`.",
    "xce": "Accessor to :class:`~lnpy.ensembles.xCanonical`.",
}


docfiller = DocFiller.from_docstring(_docstrings, combine_keys="parameters").update(
    accessor=_accessor_docs
)
