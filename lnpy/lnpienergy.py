import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from skimage import segmentation

from ._docstrings import docfiller_shared
from .cached_decorators import gcached
from .lnpiseries import lnPiCollection
from .utils import get_tqdm_calc as get_tqdm
from .utils import labels_to_masks, masks_change_convention, parallel_map_func_starargs


def find_boundaries(masks, mode="thick", connectivity=None, **kws):
    """
    find boundary region for masks

    Parameters
    ----------
    masks : list of arrays of bool
        Convesions is `masks[i][index] == True`` if `index` is active for the ith mask
    mode : str, default="thick"
        mode to use in :func:`skimage.segmentation.find_boundaries`
    connectivity : int, default = masks[0].ndim


    Returns
    -------
    boundaries : list of arrays of bool, optional
        If suppied, use these areas as boundaries.  Otherwise, calculate
        boundaries using `segmentation.find_boundaries`

    See Also
    --------
    skimage.segmentation.find_boundaries

    """
    if connectivity is None:
        connectivity = np.asarray(masks[0]).ndim

    return [
        segmentation.find_boundaries(m, connectivity=connectivity, mode=mode, **kws)
        for m in masks
    ]


def find_boundaries_overlap(
    masks,
    boundaries=None,
    flag_none=True,
    mode="thick",
    connectivity=None,
    method="exact",
):
    """
    Find regions where boundaries overlap

    Parameters
    ----------
    masks : list of arrays of bool
        Convesions is `masks[i][index] == True`` if `index` is active for the ith mask
    boundaries : list of arrays of bool, optional
        If suppied, use these areas as boundaries.  Otherwise, calculate
        boundaries using `find_boundaries`
    flag_none : bool, default=True
        if True, replace overlap with None if no overlap between regions
    method : str, {'approx', 'exact'}
        * approx : consider regions where boundaries overlap and in one of the two region
        * exact : consider regions where boundaries overlap

    Returns
    -------
    overlap : dict of masks
        overlap[i, j] = region of overlap between boundaries and masks in regions i and j


    See Also
    --------
    find_boundaries
    """

    n = len(masks)
    assert method in ["approx", "exact"]

    if boundaries is None:
        boundaries = find_boundaries(masks, mode=mode, connectivity=connectivity)

    assert n == len(boundaries)

    result = {}
    for i, j in itertools.combinations(range(n), 2):
        if method == "approx":
            # overlap region is where boundaries overlap, and in one of the masks
            overlap = (boundaries[i] & boundaries[j]) & (masks[i] | masks[j])
            if flag_none and not np.any(overlap):
                overlap = None
            result[i, j] = overlap
        elif method == "exact":
            boundaries_overlap = boundaries[i] & boundaries[j]
            for index, m in enumerate([masks[i], masks[j]]):
                overlap = boundaries_overlap & m
                if flag_none and not np.any(overlap):
                    overlap = None
                result[i, j, index] = overlap

    return result


@docfiller_shared
def find_masked_extrema(
    data,
    masks,
    convention="image",
    extrema="max",
    fill_val=np.nan,
    fill_arg=None,
    unravel=True,
):
    """
    Find position and value of extrema of masked data.

    Parameters
    ----------
    data : ndarray
        input data to consider
    masks : list of bool arrays
        masks for data regions
    {mask_convention}
    extrema : {{'max','min}}
        Type of extrema to calculate
    fill_val : scalar, default=nan
        Value to fill for `out_arg` for empty `mask`
    fill_arg : scalar, optional
        Value to fill for `out_val` for empty `mask`
    unravel : bool, default=True
        If True, unravel flat index to multi dimensional index.


    Returns
    -------
    out_arg : list of indices
        index of extrema, one for each `mask`
    out_val : ndarray
        value of extrema, one for each `mask`
    """

    if extrema == "max":
        func = np.argmax
    elif extrema == "min":
        func = np.argmin
    else:
        raise ValueError('extrema must be on of {"min", "max}')

    masks = masks_change_convention(masks, convention, "image")

    data_flat = data.reshape(-1)
    positions_flat = np.arange(data.size)

    out_val = []
    out_arg = []

    for mask in masks:

        if mask is None or not np.any(mask):
            arg = fill_arg
            val = fill_val
        else:
            mask_flat = mask.reshape(-1)
            arg = positions_flat[mask_flat][func(data_flat[mask_flat])]
            val = data_flat[arg]

            if unravel:
                arg = np.unravel_index(arg, data.shape)

        out_arg.append(arg)
        out_val.append(val)

    out_val = np.array(out_val)

    return out_arg, out_val


@docfiller_shared
def merge_regions(
    w_tran,
    w_min,
    masks,
    nfeature_max=None,
    efac=1.0,
    force=True,
    convention="image",
    warn=True,
):
    r"""
    Merge labels where free energy energy barrier < efac.

    The scaled free energy :math:`w = \beta f = - \ln \Pi ` is analized.
    Anywhere where :math:`\Delta w = \text{{w_tran}} - \text{{w_min}} < \text{{efac}}` will be merged into a single phase.  Here,
    ``w_trans`` is the transition energy between phases (i.e., the minimum value of `lnPi`` between regions`) and ``w_min`` is the minimum energy (the maximum `lnPi`) for a phase.



    Parameters
    ----------
    w_tran : array
        Shape of array is ``(n, n)``, where ``n`` is the number of unique regions/phases.
    This is the transitional free energy ()
    w_min : array
        Shape of array is ``(n, 1)``.
    masks : sequence of bool arrays
        Masks for each region using `convention`
    nfeature_max : int
        maximum number of features/phases to allow
    efac : float, default=0.5
        Wnergy difference to merge on. When ``w_trans[i, j] - w_min[i] < efac``, phases
        ``i`` and ``j`` will be merged together.
    force : bool, default=True
        if True, then keep going until nfeature <= nfeature_max
        even if min_val > efac.
    {mask_convention}
    warn : bool, default=True
        if True, give warning messages

    Returns
    -------
    masks : list of bool arrays
        output masks using `convention`
    w_trans : array
        transition energy for new masks
    w_min : array
        free energy minima of new masks
    """

    nfeature = len(masks)
    if nfeature_max is None:
        nfeature_max = nfeature

    w_tran = np.array(w_tran, copy=True)
    w_min = np.array(w_min, copy=True)

    # keep track of keep/kill
    mapping = {i: msk for i, msk in enumerate(masks)}
    for cnt in range(nfeature):
        # number of finite minima
        nfeature = len(mapping)

        de = w_tran - w_min

        min_arg = np.unravel_index(np.nanargmin(de), de.shape)
        min_val = de[min_arg]

        if min_val > efac:
            if not force:
                if warn and nfeature > nfeature_max:
                    warnings.warn(
                        "min_val > efac, but still too many phases",
                        Warning,
                        stacklevel=2,
                    )
                break
            elif nfeature <= nfeature_max:
                break

        idx_keep, idx_kill = min_arg
        # keep the one with lower energy
        if w_min[idx_keep, 0] > w_min[idx_kill, 0]:
            idx_keep, idx_kill = idx_kill, idx_keep

        # transition from idx_keep to any other phase equals the minimum transition
        # from either idx_keep or idx_kill to that other phase
        new_tran = w_tran[[idx_keep, idx_kill], :].min(axis=0)
        new_tran[idx_keep] = np.inf
        w_tran[idx_keep, :] = w_tran[:, idx_keep] = new_tran

        # get rid of old one
        w_tran[idx_kill, :] = w_tran[:, idx_kill] = np.inf

        # new mask
        mapping[idx_keep] |= mapping[idx_kill]
        del mapping[idx_kill]

    # from mapping create some new stuff
    # new w/de
    idx_min = list(mapping.keys())
    w_min = w_min[idx_min]

    idx_tran = np.ix_(*(idx_min,) * 2)
    w_tran = w_tran[idx_tran]

    # get masks
    masks = [mapping[i] for i in idx_min]

    # optionally convert image
    masks = masks_change_convention(masks, True, convention)

    return masks, w_tran, w_min


@docfiller_shared
class wFreeEnergy(object):
    r"""
    Analysis of local free energy :math:`w = \beta f = - \ln \Pi`.

    Parameters
    ----------
    data : ndarray
        lnPi data
    masks : sequence of arrays of bool
        `masks[i]` is the mask indicating the `ith` phase.  See `convention`.
    {mask_convention}
    connectivity : int, optional
        connectivity parameter for boundary construction
    index : sequence of ints, optional
        Optional index to apply to phases.
        Not yet fully supported.

    Notes
    -----
    this class uses the 'image' convention for masks.  That is
    where `mask == True` indicates locations in the phase/region,
    and `mask == False` indicates locations not in the phase/region.
    This is opposite the convention used in :mod:`numpy.ma`.
    """

    def __init__(self, data, masks, convention="image", connectivity=None, index=None):
        self.data = np.asarray(data)

        # make sure masks in image convention
        self.masks = masks_change_convention(masks, convention, "image")

        if index is None:
            index = np.arange(self.nfeature)
        self.index = index

        if connectivity is None:
            connectivity = self.data.ndim
        self.connectivity = connectivity

    @property
    def nfeature(self):
        """number of features/regions/phases"""
        return len(self.masks)

    @classmethod
    @docfiller_shared
    def from_labels(
        cls,
        data,
        labels,
        connectivity=None,
        features=None,
        include_boundary=False,
        **kwargs
    ):
        """
        create wFreeEnergy from labels

        Parameters
        ----------
        data : ndarray
            lnPi data
        {labels}
        connectivity : int, optional
            connectivity parameters
        features : array-like, optional
            list of features to extract from labels.  Note that returned
            mask[i] corresponds to labels == feature[i].
        include_boundary : bool, default=False
            if True, include boundary regions in output mask
        **kwargs
            Extra arguments to :func:`lnpy.utils.labels_to_masks`

        Returns
        -------
        out : wFreeEnergy instance

        See Also
        --------
        lnPi.utils.labels_to_masks
        """
        masks, features = labels_to_masks(
            labels,
            features=features,
            convention="image",
            include_boundary=include_boundary,
            **kwargs
        )
        return cls(data=data, masks=masks, connectivity=connectivity)

    @gcached()
    def _data_max(self):
        """
        for lnPi data, find absolute argmax and max
        """
        return find_masked_extrema(self.data, self.masks)

    @gcached(prop=False)
    def _boundary_max(self, method="exact"):
        """
        find argmax along boundaries of regions.
        Corresponds to argmin(w)

        if method == 'exact', then find the boundary of each region
        and find max.  then find min of those maxes.
        """
        overlap = find_boundaries_overlap(
            self.masks,
            mode="thick",
            connectivity=self.connectivity,
            flag_none=True,
            method=method,
        )
        argmax, valmax = find_masked_extrema(
            self.data,
            overlap.values(),
            fill_val=np.nan,
            fill_arg=None,
        )

        # unpack output
        out_arg = {}
        out_max = np.full((self.nfeature,) * 2, dtype=float, fill_value=np.nan)
        if method == "approx":
            for (i, j), arg, val in zip(overlap.keys(), argmax, valmax):
                out_max[i, j] = out_max[j, i] = val
                out_arg[i, j] = arg

        elif method == "exact":
            # attach keys to argmax, valmax
            argmax = dict(zip(overlap.keys(), argmax))
            valmax = dict(zip(overlap.keys(), valmax))

            # first get unique keys:
            keys = [(i, j) for i, j, _ in overlap.keys()]
            keys = list(set(keys))

            for (i, j) in keys:
                vals = [valmax[i, j, index] for index in range(2)]
                # take min value of maxes
                if np.all(np.isnan(vals)):
                    out_arg[i, j] = None
                else:
                    idx_min = np.nanargmin(vals)
                    out_arg[i, j] = argmax[i, j, idx_min]
                    out_max[i, j] = out_max[j, i] = valmax[i, j, idx_min]
        return out_arg, out_max

    @property
    def w_min(self):
        """Minimum value of `w` (max `lnPi`) in each phase/region."""
        return -self._data_max[1][:, None]

    @property
    def w_argmin(self):
        """locations of the minimum of `w` in each phase/region"""
        return self._data_max[0]

    @property
    def w_tran(self):
        """Minimum value of `w` (max `lnPi`) in the boundary between phases.

        `w_tran[i, j]` is the transition energy between phases `i` ans `j`.
        """
        return np.nan_to_num(-self._boundary_max()[1], nan=np.inf)

    @property
    def w_argtran(self):
        """location of `w_tran`"""
        return self._boundary_max()[0]

    @gcached()
    def delta_w(self):
        """Transition energy ``delta_w[i, j] = w_tran[i, j] - w_min[i]``."""
        return self.w_tran - self.w_min

    def merge_regions(
        self,
        nfeature_max=None,
        efac=1.0,
        force=True,
        convention="image",
        warn=True,
    ):
        """
        Merge labels where free energy energy barrier < efac.

        Interface to :func:`merge_regions`

        Parameters
        ----------
        nfeature_max : int
            maximum number of features/phases to allow
        efac : float, default=0.5
            Wnergy difference to merge on. When ``w_trans[i, j] - w_min[i] < efac``, phases
            ``i`` and ``j`` will be merged together.
        force : bool, default=True
            if True, then keep going until nfeature <= nfeature_max
            even if min_val > efac.
        {mask_convention}
        warn : bool, default=True
            if True, give warning messages

        Returns
        -------
        masks : list of bool arrays
            output masks using `convention`
        w_trans : array
            transition energy for new masks
        w_min : array
            free energy minima of new masks
        """

        return merge_regions(
            w_tran=self.w_tran,
            w_min=self.w_min,
            masks=self.masks,
            nfeature_max=nfeature_max,
            efac=efac,
            force=force,
            convention=convention,
            warn=warn,
        )


def _get_w_data(index, w):
    w_min = pd.Series(w.w_min[:, 0], index=index, name="w_min")
    w_argmin = pd.Series(w.w_argmin, index=w_min.index, name="w_argmin")

    w_tran = (
        pd.DataFrame(
            w.w_tran,
            index=index,
            columns=index.get_level_values("phase").rename("phase_nebr"),
        )
        .stack()
        .rename("w_tran")
    )

    # get argtrans values for each index
    index_map = {idx: i for i, idx in enumerate(index.get_level_values("phase"))}
    v = w.w_argtran

    argtran = []
    for idxs in zip(
        *[w_tran.index.get_level_values(_) for _ in ["phase", "phase_nebr"]]
    ):
        i, j = [index_map[_] for _ in idxs]

        if (i, j) in v:
            val = v[i, j]
        elif (j, i) in v:
            val = v[j, i]
        else:
            val = None
        argtran.append(val)

    w_argtran = pd.Series(argtran, index=w_tran.index, name="w_argtran")

    return {
        "w_min": w_min,
        "w_tran": w_tran,
        "w_argmin": w_argmin,
        "w_argtran": w_argtran,
    }  # [index_map, w.w_argtran]}


class wFreeEnergyCollection(object):
    r"""
    Calculate the transition free energies for a :class:`lnpy.lnPiCollection`.

    :math:`w(N) = \beta f(N) = - \ln \Pi(N)`

    Parameters
    ----------
    parent : lnPiCollection

    Notes
    -----
    An instance of :class:`wFreeEnergyCollection` is normally created from the accessor :meth:`lnpy.lnPiCollection.wfe`
    """

    def __init__(self, parent):
        self._parent = parent
        self._use_joblib = getattr(self._parent, "_use_joblib", False)

    def _get_items_ws(self):
        indexes = []
        ws = []
        for meta, phases in self._parent.groupby_allbut("phase"):
            indexes.append(phases.index)
            masks = [x.mask for x in phases.values]
            ws.append(
                wFreeEnergy(data=phases.iloc[0].data, masks=masks, convention=False)
            )
        return indexes, ws

    @gcached()
    def _data(self):
        indexes, ws = self._get_items_ws()
        seq = get_tqdm(zip(indexes, ws), total=len(ws), desc="wFreeEnergyCollection")
        out = parallel_map_func_starargs(
            _get_w_data, items=seq, use_joblib=self._use_joblib, total=len(ws)
        )

        result = {key: pd.concat([x[key] for x in out]) for key in out[0].keys()}

        return result

    @property
    def w_min(self):
        """Minimum energy (maximum `lnPi`) for a given region/phase"""
        return self._data["w_min"]

    @property
    def w_tran(self):
        """Minimum energy (maximum `lnPi`) at boundary between phases"""
        return self._data["w_tran"]

    @property
    def w_argmin(self):
        """location of :attr:`w_min`"""
        return self._data["w_argmin"]

    @property
    def w_argtran(self):
        """location of :attr:`w_tran`"""
        return self._data["w_argtran"]

    @property
    def dw(self):
        """Series representation of `dw = w_tran - w_min`"""
        return (self.w_tran - self.w_min).rename("delta_w")

    @property
    def dwx(self):
        """xarray representation of :attr:`dw`"""
        return self.dw.to_xarray()

    def get_dwx(self, idx, idx_nebr=None):
        """
        helper function to get the change in energy from
        phase idx to idx_nebr.

        Parameters
        ----------
        idx : int
            phase index to consider transitions from
        idx_nebr : int or list, optional
            if supplied, consider transition from idx to idx_nebr or minimum of all element in idx_nebr.
            Default behavior is to return minimum transition from idx to all other neighboring regions

        Returns
        -------
        dw : DataArray
            Transition energy from `idx` to `idx_nebr`
            - if only phase idx exists, dw = np.inf
            - if idx does not exists, dw = 0.0 (no barrier between idx and anything else)
            - else min of transition for idx to idx_nebr
        """

        delta_w = self.dwx

        # reindex so that has idx in phase
        reindex = delta_w.indexes["phase"].union(pd.Index([idx], name="phase"))
        delta_w = delta_w.reindex(phase=reindex, phase_nebr=reindex)

        # much simpler
        if idx_nebr is None:
            delta_w = delta_w.sel(phase=idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            if idx not in idx_nebr:
                idx_nebr.append(idx)
            nebrs = delta_w.indexes["phase_nebr"].intersection(idx_nebr)
            delta_w = delta_w.sel(phase=idx, phase_nebr=nebrs)

        out = delta_w.min("phase_nebr").fillna(0.0)
        return out

    def get_dw(self, idx, idx_nebr=None):
        """Sereis version of :meth:`get_dwx`"""
        return self.get_dwx(idx, idx_nebr).to_series()


# @lnPiCollection.decorate_accessor("wfe_phases")
class wFreeEnergyPhases(wFreeEnergyCollection):
    """
    Stripped down version of :class:`wFreeEnergyCollection` for single phase grouping.

    This should be used for a collection of lnPi that is at a single state point, with multiple phases.

    Parameters
    ----------
    parent : lnPiCollection

    Notes
    -----
    This is accessed through :attr:`lnpy.lnPiCollection.wfe_phases`

    """

    @gcached()
    def dwx(self):
        index = list(self._parent.index.get_level_values("phase"))
        masks = [x.mask for x in self._parent]
        w = wFreeEnergy(data=self._parent.iloc[0].data, masks=masks, convention=False)

        dw = w.w_tran - w.w_min
        dims = ["phase", "phase_nebr"]
        coords = dict(zip(dims, [index] * 2))
        return xr.DataArray(dw, dims=dims, coords=coords)

    @gcached()
    def dw(self):
        """Series representation of delta_w"""
        return self.dwx.to_series()

    def get_dw(self, idx, idx_nebr=None):
        dw = self.dwx
        index = dw.indexes["phase"]

        if idx not in index:
            return 0.0
        elif idx_nebr is None:
            nebrs = index.drop(idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            nebrs = [x for x in idx_nebr if x in index]

        if len(nebrs) == 0:
            return np.inf
        return dw.sel(phase=idx, phase_nebr=nebrs).min("phase_nebr").values


@lnPiCollection.decorate_accessor("wfe")
def wfe_accessor(parent):
    """
    Accessor to :class:`~lnpy.wFreeEnergyCollection` from `self.wfe`.
    """
    return wFreeEnergyCollection(parent)


@lnPiCollection.decorate_accessor("wfe_phases")
def wfe_phases_accessor(parent):
    """Accessor to :class:`~lnpy.wFreeEnergyPhases` from `self.wfe_phases`."""
    return wFreeEnergyPhases(parent)


from warnings import warn


# create alias accessors
@lnPiCollection.decorate_accessor("wlnPi")
def wlnPi_accessor(parent):
    """Deprecated accessor to :class:`~lnpy.wFreeEnergyCollection` from `self.wlnPi`.

    Alias to `self.wfe`
    """
    warn("Using `wlnPi` accessor is deprecated.  Please use `wfe` accessor instead")
    return parent.wfe


@lnPiCollection.decorate_accessor("wlnPi_single")
def wlnPi_single_accessor(parent):
    """Deprecated accessor to :class:`~lnpy.wFreeEnergyPhases` from `self.wlnPi_single`.

    Alias to `self.wfe_single`
    """
    warn("Using `wlnPi_single is deprecated.  Please use `self.wfe_phases` instead")
    return parent.wfe_phases
