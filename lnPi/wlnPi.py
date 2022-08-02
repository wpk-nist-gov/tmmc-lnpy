import itertools
import warnings

import numpy as np
from skimage import segmentation

from .cached_decorators import gcached
from .utils import labels_to_masks, masks_change_convention


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

    def _find_boundaries(self, idx):
        """
        Find boundaries between each regions
        """
        return segmentation.find_boundaries(
            self.masks[idx], connectivity=self.connectivity, mode="thick"
        )

    # @gcached() # no need to cache
    @property
    def _boundaries(self):
        """boundary of each label"""
        return [self._find_boundaries(i) for i in self.index]

    # @gcached()
    @property
    def _boundaries_overlap(self):
        """overlap of boundaries"""
        boundaries = {}
        for i, j in itertools.combinations(self.index, 2):
            # instead of using foreground, maker sure that the boundary
            # is contained in eigher region[i] or region[j]
            overlap = (
                # overlap of boundary
                (self._boundaries[i] & self._boundaries[j])
                # overlap with with union of region
                & (self.masks[i] | self.masks[j])
            )

            if overlap.sum() == 0:
                overlap = None
            boundaries[i, j] = overlap
        return boundaries

    def _get_masked_array(self, mask, convention="image"):
        mask = masks_change_convention(mask, convention, "masked")
        return np.ma.MaskedArray(self.data, mask=mask)

    @gcached()
    def argw_min(self):
        argmax = []
        for mask in self.masks:
            m = self._get_masked_array(mask, convention="image")
            # create masked array
            idx = m.argmax()
            # unravel
            idx = np.unravel_index(idx, self.data.shape)
            argmax.append(idx)
        return argmax

    @gcached()
    def argw_tran(self):
        out = {}
        for (i, j), boundary in self._boundaries_overlap.items():
            if boundary is None:
                val = np.inf
            else:
                m = self._get_masked_array(boundary, convention="image")
                # want min of -lnPi, so get max of lnPi
                val = np.unravel_index(m.argmax(), self.data.shape)
            out[i, j] = val
        return out

    @property
    def w_min2(self):
        w_min = np.array([-self.data[k] for k in self.argw_min])
        return w_min.reshape(-1, 1)

    @property
    def w_tran2(self):
        out = np.full((self.nfeature,) * 2, dtype=float, fill_value=np.inf)

        for phase_idx, val_idx in self.argw_tran.items():
            val = -self.data[val_idx]
            out[phase_idx] = val

            if len(phase_idx) > 1:
                out[phase_idx[-1::-1]] = val

        return out

    @property
    def _w_min(self):
        return -np.array([self.data[msk].max() for msk in self.masks])

    @property
    def w_min(self):
        return self._w_min.reshape(-1, 1)

    @gcached()
    def w_tran(self):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        out = np.full((self.nfeature,) * 2, dtype=float, fill_value=np.inf)

        for (i, j), boundary in self._boundaries_overlap.items():
            # label to zero based
            if boundary is None:
                val = np.inf
            else:
                val = (-self.data[boundary]).min()

            out[i, j] = out[j, i] = val
        return np.array(out)

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
