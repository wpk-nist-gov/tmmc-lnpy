import itertools
from collections import Iterable

import numpy as np

from scipy import ndimage as ndi
from skimage import feature, morphology, segmentation
#from skimage.morphology import watershed
#from skimage.segmentation import find_boundaries
import bottleneck

from .cached_decorators import gcached

def peak_local_max_adaptive(data,
                            labels=None,
                            min_distance=[5, 10, 15, 20, 25],
                            threshold_rel=0.00,
                            threshold_abs=0.2,
                            num_peaks_max=None,
                            indices=True,
                            **kwargs):
    """
    find local max with fall backs min_distance and filter

    Parameters
    ----------
    data : image to analyze
    mask : optional mask (True means include, False=exclude)
    min_distance : int or iterable (Default 15)
        min_distance parameter to self.peak_local_max.
        if min_distance is iterable, if num_phase>num_phase_max, try next
    num_peaks_max : int (Default None)
        max number of maxima to find.
    **kwargs : extra arguments to peak_local_max

    Returns
    -------
    out : tuple of ndarrays
        indices of self where local max

    out_info : tuple (min_distance,smooth)
        min_distance used and bool indicating if smoothing was used
    """

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)

    kwargs = dict(dict(exclude_border=False), **kwargs)

    for md in min_distance:
        idx = feature.peak_local_max(
            data, min_distance=md,
            labels=labels,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            indices=True, **kwargs)

        n = len(idx)
        if n <= num_peaks_max:
            idx = tuple(idx.T)
            if indices:
                return idx
            else:
                out = np.zeros_like(data, dtype=np.bool)
                out[idx] = True
                return out

    #if got here than error
    raise RuntimeError('%i maxima found greater than %i' %
                        (n, num_peaks_max))



class Segmenter(object):
    """
    Segment lnpi
    """
    def __init__(self, min_distance=[5,10], peak_kws=None, watershed_kws=None):

        if peak_kws is None:
            peak_kws = {}
        peak_kws.update(indices=False, min_distance=min_distance)

        if watershed_kws is None:
            watershed_kws = {}
        self._peak_kws = peak_kws
        self._watershed_kws = watershed_kws

    def peaks(self, data, mask, num_peaks_max=None, as_marker=True, connectivity=None, **kwargs):
        kwargs = dict(self._peak_kws, **kwargs)
        if num_peaks_max is not None:
            kwargs['num_peaks_max'] = num_peaks_max
        out = peak_local_max_adaptive(data, labels=mask, **kwargs)
        if as_marker:
            out = morphology.label(out, connectivity=connectivity)
        return out


    def watershed(self, data, markers, mask, connectivity=None, **kwargs):
        if connectivity is None:
            connectivity = data.ndim
        kwargs = dict(self._watershed_kws, connectivity=connectivity, *kwargs)
        return morphology.watershed(-data, markers=markers, mask=mask, **kwargs)


    def segment_lnpi(self, lnpi, connectivity=None):

        markers = self.peaks(lnpi.data, mask=~lnpi.mask, connectivity=connectivity)
        labels = self.watershed(lnpi.data, markers=markers, mask=~lnpi.mask, connectivity=connectivity)

        return labels




class Merger(object):
    """
    given lnpi, try to merge chunks
    """
    def __init__(self, data, labels, index=None, connectivity=None, nfeature=None, foreground=None):

        if index is None:
            if nfeature is None:
                index = np.unique(labels)
                index = index[index > 0]
                nfeature = index.max()
                assert np.all(index == np.arange(1, nfeature+1))
            else:
                index = np.arange(1, nfeature + 1)


        self._data = data
        self._labels = labels - 1
        self._index = index - 1
        self._nfeature = nfeature

        if foreground is None:
            foreground = self._labels >= 0
        self._foreground = foreground

        if connectivity is None:
            connectivity = data.ndim
        self._connectivity = connectivity


    def _find_boundaries(self, idx):
        return segmentation.find_boundaries(self._masks[idx], connectivity= self._connectivity, mode='thick')

    @gcached()
    def _masks(self):
        return [self._labels == i for i in self._index]

    @gcached()
    def _boundaries(self):
        """boundary of each label"""
        return [self._find_boundaries(i) for i in self._index]

    @gcached()
    def _boundaries_overlap(self):
        """overlap of boundaries"""
        boundaries = {}
        for i,j in itertools.combinations(self._index, 2):
            overlap = (
                self._boundaries[i] &
                self._boundaries[j] &
                self._foreground)

            if overlap.sum() == 0:
                overlap = None
            boundaries[i, j] = overlap
        return boundaries

    @gcached()
    def w_min(self):
        return -np.array([self._data[msk].max() for msk in self._masks]).reshape(-1, 1)

    @gcached()
    def w_tran(self, **kwargs):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        out = np.full((self._nfeature,)*2, dtype=np.float, fill_value=np.inf)

        for (i,j), boundary in self._boundaries_overlap.items():
            # label to zero based
            if boundary is None:
                val = np.inf
            else:
                val = -(self._data[boundary]).max()

            out[i, j] = out[j, i] = val
        return np.array(out)

    @gcached()
    def _delta_w(self):
        return self.w_tran - self.w_min

    def merge_labels(self, nfeature_max=None, efac=1.0, force=True, **kwargs):

        if nfeature_max is None:
            nfeature_max = self._nfeature

        w_tran = self.w_tran.copy()
        w_min  = self.w_min.copy()

        # keep track of keep/kill
        #mapping[keep] = [keep, merge_in_1, ...]
        mapping = {i : [i] for i in self._index}

        for cnt in range(self._nfeature):
            # number of finite minima
            nfeature = np.isfinite(w_min).sum()

            de = w_tran - w_min
            min_val = np.nanmin(de)

            if min_val > efac:
                if not force:
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


            mapping[idx_keep] += mapping[idx_kill]
            del mapping[idx_kill]

        # from mapping create some new stuff
        # new w/de
        idx = list(mapping.keys())
        w_min = w_min[idx]

        idx = np.ix_(*(idx,)*2)
        w_tran = w_tran[idx]


        # new labels
        new_labels = np.zeros_like(self._labels)
        for _feature,(i, _labels) in enumerate(mapping.items()):
            for _label in _labels:
                new_labels[self._labels == _label] = _feature + 1

        return new_labels, w_tran, w_min

