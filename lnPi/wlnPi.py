import itertools
import warnings

import numpy as np
from skimage import segmentation

from .cached_decorators import gcached
from .utils import labels_to_masks, masks_change_convention


def find_boundaries(masks, mode="thick", connectivity=None, **kws):
    """
    find boundary region for masks

    Parameters
    ----------
    masks : list of arrays of bool
        Convesions is `masks[i][index] == True`` if `index` is active for the ith mask
    mode : str, default="thick"
        mode to use in `segmentation.find_boundaries`
    connectivity : int, default = masks[0].ndim


    Returns
    -------
    boundaries : list of arrays of bool, optional
        If suppied, use these areas as boundaries.  Otherwise, calculate
        boundaries using `segmentation.find_boundaries`

    """
    if connectivity is None:
        connectivity = np.asarray(masks[0]).ndim

    return [
        segmentation.find_boundaries(m, connectivity=connectivity, mode=mode, **kws)
        for m in masks
    ]


def find_boundaries_overlap(
    masks, boundaries=None, flag_none=True, mode="thick", connectivity=None
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

    Returns
    -------
    overlap : dict of masks
        overlap[i, j] = region of overlap between boundaries and masks in regions i and j
    """

    n = len(masks)

    if boundaries is None:
        boundaries = find_boundaries(masks, mode=mode, connectivity=connectivity)

    assert n == len(boundaries)

    result = {}
    for i, j in itertools.combinations(range(n), 2):
        # overlap region is where boundaries overlap, and in one of the masks
        overlap = (boundaries[i] & boundaries[j]) & (masks[i] | masks[j])
        if flag_none and not np.any(overlap):
            overlap = None
        result[i, j] = overlap
    return result


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
    find position and value of extrema of masked data
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

        if not np.any(mask):
            arg = fill_arg
            val = fill_arg
        else:
            mask_flat = mask.reshape(-1)
            arg = positions_flat[mask_flat][func(data_flat[mask_flat])]
            val = data_flat[arg]

        out_arg.append(arg)
        out_val.append(val)

    out_val = np.array(out_val)

    if unravel:
        out_arg = [np.unravel_index(idx, data.shape) for idx in out_arg]

    return out_arg, out_val


class FreeEnergylnPi(object):
    """
    find/merge the transition energy between minima and barriers
    in lnPi

    here we define the free energy w = betaW = -ln(Pi)

    NOTE : this class used the image convention that
    mask == True indicates that the region includes the feature.
    This is oposite the masked array convension, where mask==True implies that region is masked out.
    """

    def __init__(self, data, masks, convention="image", connectivity=None, index=None):
        """
        Parameters
        ----------
        data : array
            lnPi data
        masks : list of arrays
            masks[i] == True where feature exists
        convention : str or bool
            convention of masks
        connectivity : int, optional
            connectivity parameter for boundary construction
        """
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
        return len(self.masks)

    @classmethod
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
        create FreeEnergylnPi from labels
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

    @gcached()
    def _boundary_max(self):
        """
        find argmax along boundaries of regions.
        Corresponds to argmin(w)
        """
        overlap = find_boundaries_overlap(
            self.masks, mode="thick", connectivity=self.connectivity, flag_none=True
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

        for (i, j), arg, val in zip(overlap.keys(), argmax, valmax):
            out_max[i, j] = out_max[j, i] = val
            out_arg[i, j] = arg

        return out_arg, out_max

    @property
    def w_min(self):
        return -self._data_max[1][:, None]

    @property
    def w_argmin(self):
        return self._data_max[0]

    @property
    def w_tran(self):
        return np.nan_to_num(-self._boundary_max[1], nan=np.inf)

    @property
    def w_argtran(self):
        return self._boundary_max[0]

    @gcached()
    def delta_w(self):
        """
        -beta (lnPi[transition] - lnPi[max])
        """
        return self.w_tran - self.w_min

    def merge_regions(
        self,
        nfeature_max=None,
        efac=1.0,
        force=True,
        convention="image",
        warn=True,
        **kwargs
    ):
        """
        merge labels where free energy energy barrier < efac.

        Parameters
        ----------
        nfeature_max : int
            maximum number of features
        efac : float, default=0.5
            energy difference to merge on
        force : bool, default=True
            if True, then keep going until nfeature <= nfeature_max
            even if min_val > efac
        convention : str or bool, default=True
            convention of output masks
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

        if nfeature_max is None:
            nfeature_max = self.nfeature

        w_tran = self.w_tran.copy()
        w_min = self.w_min.copy()

        # keep track of keep/kill
        # mapping[keep] = [keep, merge_in_1, ...]
        # mapping = {i : [i] for i in self._index}
        mapping = {i: msk for i, msk in enumerate(self.masks)}
        for cnt in range(self.nfeature):
            # number of finite minima
            nfeature = len(mapping)
            # nfeature = np.isfinite(w_min).sum()

            de = w_tran - w_min
            min_val = np.nanmin(de)

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

            idx_keep, idx_kill = [x[0] for x in np.where(de == min_val)]

            # keep the one with lower energy
            if w_min[idx_keep] > w_min[idx_kill]:
                idx_keep, idx_kill = idx_kill, idx_keep

            # idx[0] and idx[1] merge together
            # arbitrarily bick idx[0] to keep and idx[1] to kill

            # transition from idx_keep to any other phase equals the minimum transition
            # from either idx_keep or idx_kill to that other phase
            new_tran = w_tran[[idx_keep, idx_kill], :].min(axis=0)
            new_tran[idx_keep] = np.inf

            w_tran[idx_keep, :] = w_tran[:, idx_keep] = new_tran
            # get rid of old one
            w_tran[idx_kill, :] = w_tran[:, idx_kill] = np.inf

            # mapping[idx_keep] += mapping[idx_kill]
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
