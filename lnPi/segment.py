import itertools
from collections import Iterable

import numpy as np

from scipy import ndimage as ndi
from skimage import feature, morphology, segmentation
import bottleneck

from .cached_decorators import gcached
from .utils import (
    mask_change_convention, masks_change_convention, labels_to_masks, masks_to_labels
)

import warnings

from .core import Phases


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
    labels : optional mask (True means include, False=exclude)
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
            data,
            min_distance=md,
            labels=labels,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            indices=True,
            **kwargs)

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
    raise RuntimeError('%i maxima found greater than %i' % (n, num_peaks_max))


class Segmenter(object):
    """
    Segment lnpi
    """

    def __init__(self, min_distance=[1, 5, 10, 15, 20], peak_kws=None,
                 watershed_kws=None):

        if peak_kws is None:
            peak_kws = {}
        peak_kws.update(indices=False, min_distance=min_distance)

        if watershed_kws is None:
            watershed_kws = {}
        self._peak_kws = peak_kws
        self._watershed_kws = watershed_kws

    def peaks(self,
              data,
              mask,
              num_peaks_max=None,
              as_marker=True,
              connectivity=None,
              **kwargs):
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
        return morphology.watershed(
            -data, markers=markers, mask=mask, **kwargs)

    def segment_lnpi(self, lnpi, num_peaks_max=None, connectivity=None):

        markers = self.peaks(
            lnpi.data, mask=~lnpi.mask, num_peaks_max=num_peaks_max, connectivity=connectivity)
        labels = self.watershed(
            lnpi.data,
            markers=markers,
            mask=~lnpi.mask,
            connectivity=connectivity)

        return labels


class FreeEnergylnPi(object):
    """
    find/merge the transition energy between minima and barriers
    in lnPi

    NOTE : this class used the image convension that
    mask == True indicates that the region includes the feature.  This is oposite the masked array convension, where mask==True implies that region is masked out.
    """

    def __init__(self,
                 data,
                 masks,
                 convention='image',
                 background=None,
                 connectivity=None,
                 index=None):
        """
        Parameters
        ----------
        data : array
            lnPi data
        masks : list of arrays
            masks[i] == True where feature exists
        convention : str or bool
            convention of masks
        background : bool array, optional
            background == False where features can exist
        connectivity : int, optional
            connectivity parameter for boundary construction
        """
        self._data = np.asarray(data)

        # make sure masks in image convention
        self._masks = masks_change_convention(masks, convention, 'image')

        self._nfeature = len(self._masks)
        if index is None:
            index = np.arange(self._nfeature)
        self._index = index

        if background is None:
            background = np.zeros_like(self._data, dtype=np.bool)
        self._background = np.array(background, dtype=np.bool)
        self._foreground = ~self._background

        if connectivity is None:
            connectivity = data.ndim
        self._connectivity = connectivity

    @classmethod
    def from_labels(cls,
                    data,
                    labels,
                    background,
                    connectivity=None,
                    features=None,
                    include_boundary=False,
                    **kwargs):
        """
        create FreeEnergylnPi from labels
        """
        masks, features = labels_to_masks(
            labels,
            features=features,
            convention='image',
            include_boundary=include_boundary,
            **kwargs)
        return cls(
            data=data,
            masks=masks,
            background=background,
            connectivity=connectivity)

    def _find_boundaries(self, idx):
        return segmentation.find_boundaries(
            self._masks[idx], connectivity=self._connectivity, mode='thick')

    @gcached()
    def _boundaries(self):
        """boundary of each label"""
        return [self._find_boundaries(i) for i in self._index]

    @gcached()
    def _boundaries_overlap(self):
        """overlap of boundaries"""
        boundaries = {}
        for i, j in itertools.combinations(self._index, 2):
            overlap = (
                self._boundaries[i] & self._boundaries[j] & self._foreground)

            if overlap.sum() == 0:
                overlap = None
            boundaries[i, j] = overlap
        return boundaries

    @gcached()
    def w_min(self):
        return -np.array([self._data[msk].max()
                          for msk in self._masks]).reshape(-1, 1)

    @gcached()
    def w_tran(self, **kwargs):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        out = np.full(
            (self._nfeature, ) * 2, dtype=np.float, fill_value=np.inf)

        for (i, j), boundary in self._boundaries_overlap.items():
            # label to zero based
            if boundary is None:
                val = np.inf
            else:
                val = -(self._data[boundary]).max()

            out[i, j] = out[j, i] = val
        return np.array(out)

    @gcached()
    def delta_w(self):
        """
        -beta (lnPi[transition] - lnPi[max])
        """
        return self.w_tran - self.w_min

    def merge_regions(self,
                      nfeature_max=None,
                      efac=1.0,
                      force=True,
                      convention='image',
                      warn=True,
                      **kwargs):
        """
        merge labels where free energy energy barrier < efac.

        Parameters
        ----------
        nfeature_max : int
            maximum number of features
        efac : float, default=1.0
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
            nfeature_max = self._nfeature

        w_tran = self.w_tran.copy()
        w_min = self.w_min.copy()

        # keep track of keep/kill
        #mapping[keep] = [keep, merge_in_1, ...]
        #mapping = {i : [i] for i in self._index}
        mapping = {i: msk for i, msk in enumerate(self._masks)}
        for cnt in range(self._nfeature):
            # number of finite minima
            nfeature = np.isfinite(w_min).sum()

            de = w_tran - w_min
            min_val = np.nanmin(de)

            if min_val > efac:
                if not force:
                    if nfeature > nfeature_max:
                        warnings.warn(
                            'min_val > efac, but still too many phases',
                            Warning,
                            stacklevel=2)
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

        idx_tran = np.ix_(*(idx_min, ) * 2)
        w_tran = w_tran[idx_tran]

        # add in background
        masks = [mapping[i] & self._foreground for i in idx_min]

        # optionally convert image
        masks = masks_change_convention(masks, True, convention)

        return masks, w_tran, w_min



# class to add FreeEnergylnPi to Phases
from .core import Phases, CollectionPhases

CollectionPhases.register_listaccessor('wlnPi')

@Phases.decorate_accessor('wlnPi')
class wlnPi(FreeEnergylnPi):
    def __init__(self, phases):
        self._phases = phases
        base = self._phases[0]
        masks = [x.mask for x in self._phases]
        background = np.logical_and.reduce(masks)
        super(wlnPi, self).__init__(data=base.data, masks=masks, convention=False, background=background)



class PhaseCreator(object):
    """
    Helper class to create phases
    """

    def __init__(self, nmax, nmax_peak=None,
                 segmenter=None, segment_kws=None,
                 tag_phases=None, phases_class=Phases,
                 FreeEnergylnPi_kws=None, merge_kws=None):
        """
        Parameters
        ----------
        ref : MaskedlnPi object
        nmax : int
            number of phases to construct
       nmax_peak : int, optional
            if specified, the allowable number of peaks to locate.
            This can be useful for some cases.  These phases will be merged out at the end.
        segmenter : Segmenter object, optional
            segmenter object to create labels/masks
        Freeenergy_kws : dict, optional
            dictionary of parameters for the creation of a FreeenergylnPi object
        """

        if nmax_peak is None:
            nmax_peak = nmax * 2
        self.nmax = nmax

        if segmenter is None:
            segmenter = Segmenter()
        self.segmenter = segmenter

        self.tag_phases = tag_phases

        if phases_class is None:
            phases_class = Phases
        self.phases_class = phases_class

        if segment_kws is None:
            segment_kws = {}
        self.segment_kws = segment_kws
        self.segment_kws['num_peaks_max'] = nmax_peak

        if FreeEnergylnPi_kws is None:
            FreeEnergylnPi_kws = {}
#        FreeEnergylnPi_kws['background'] = self.ref.mask
        self.FreeEnergylnPi_kws = FreeEnergylnPi_kws

        if merge_kws is None:
            merge_kws = {}
        merge_kws = dict(merge_kws, convention=False, nfeature_max=self.nmax)
        self.merge_kws = merge_kws


    def build_phases(self, ref, mu=None, efac=1.0, nmax_peak=None, connectivity=None, reweight_kws=None, phases_output=True):

        # reweight
        if mu is not None:
            if reweight_kws is None:
                reweight_kws = {}
            ref = ref.reweight(mu, **reweight_kws)

        # labels
        kws = self.segment_kws.copy()
        if nmax_peak is not None:
            kws['num_peaks_max'] = nmax_peak
        if connectivity is not None:
            kws['connectivity'] = connectivity
        labels = self.segmenter.segment_lnpi(lnpi=ref, **kws)

        # wlnpi
        kws = dict(self.FreeEnergylnPi_kws, background=ref.mask)
        if connectivity is not None:
            kws['connectivity'] = connectivity
        wlnpi = FreeEnergylnPi.from_labels(
            data=ref.data,
            labels=labels,
            **kws)

        # merge
        kws = dict(self.merge_kws, efac=efac)
        masks, wtran, wmin = wlnpi.merge_regions(**kws)

        # list of lnpi
        lnpis = ref.list_from_masks(masks, convention=False)

        # tag phases?
        if self.tag_phases is not None:
            index = self.tag_phases(lnpis)
        else:
            index = None

        if phases_output:
            return self.phases_class(items=lnpis, index=index)
        else:
            return lnpis, index




from functools import lru_cache

@lru_cache(maxsize=10)
def get_default_PhaseCreator(nmax):
    return PhaseCreator(nmax=nmax)
