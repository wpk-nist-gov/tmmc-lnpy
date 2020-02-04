"""
routines to segment lnPi

 1. find max/peaks in lnPi
 2. segment lnPi about these peaks
 3. determine free energy difference between segments
    a. Merge based on low free energy difference
 4. combination of 1-3.
"""

import itertools
from collections import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from skimage import feature, morphology, segmentation
import bottleneck

from .cached_decorators import gcached
from .utils import labels_to_masks

import warnings
from .collectionlnpi import CollectionlnPi

from .utils import (parallel_map_func_starargs, get_tqdm_calc as get_tqdm,
                    masks_change_convention)


def peak_local_max_adaptive(data,
                            mask=None,
                            min_distance=[5, 10, 15, 20, 25],
                            threshold_rel=0.00,
                            threshold_abs=0.2,
                            num_peaks_max=None,
                            indices=True,
                            errors='warn',
                            **kwargs):
    """
    find local max with fall backs min_distance and filter

    Parameters
    ----------
    data : image to analyze
    mask : mask array of same shape as data, optional
        True means include, False=exclude.  Note that this parameter is called
        `lables` in `peaks_local_max`
    min_distance : int or iterable (Default 15)
        min_distance parameter to self.peak_local_max.
        if min_distance is iterable, if num_phase>num_phase_max, try next
    threshold_rel, threshold_abs : float
        thresholds to use in peak_local_max
    num_peaks_max : int (Default None)
        max number of maxima to find.
    indeces : bool, default=True
        if True, return indicies of peaks.
        if False, return array of ints of shape `data.shape` with peaks
        marked by value > 0.
    errors : {'ignore','raise','warn'}, default='warn'
        - If raise, raise exception if npeaks > num_peaks_max
        - If ignore, return all found maxima
        - If warn, raise warning if npeaks > num_peaks_max


    **kwargs : extra arguments to peak_local_max

    Returns
    -------
    out :
    - if indices is True, tuple of ndarrays
        indices of where local max
   """

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)
    kwargs = dict(dict(exclude_border=False), **kwargs)

    for md in min_distance:
        idx = feature.peak_local_max(data,
                                     min_distance=md,
                                     labels=mask,
                                     threshold_abs=threshold_abs,
                                     threshold_rel=threshold_rel,
                                     indices=True,
                                     **kwargs)

        n = len(idx)
        if n <= num_peaks_max:
            break

    if n > num_peaks_max:
        if errors == 'ignore':
            pass
        elif errors in ('raise', 'ignore'):
            message = '{} maxima found greater than {}'.format(
                n, num_peaks_max)
            if errors == 'raise':
                raise RuntimeError(message)
            else:
                warning.warn(message)

    idx = tuple(idx.T)
    if indices:
        return idx
    else:
        out = np.zeros_like(data, dtype=np.bool)
        out[idx] = True
        return out


class Segmenter(object):
    """
    Data segmenter:

    Methods
    -------
    peaks : find peaks of data
    watershep : watershed segementation
    segment_lnpi : helper funciton to segment lnPi
    """
    def __init__(self,
                 min_distance=[1, 5, 10, 15, 20],
                 peak_kws=None,
                 watershed_kws=None):
        """
        Parameters
        ----------
        peak_kws : dictionary
            kwargs to `peak_local_max_adaptive`
        watershed_kws : dictionary
            kwargs to `skimage.morphology.watershed`
        """

        if peak_kws is None:
            peak_kws = {}
        peak_kws.update(indices=False)
        self.peak_kws = peak_kws

        if watershed_kws is None:
            watershed_kws = {}
        self.watershed_kws = watershed_kws

    def peaks(self,
              data,
              mask=None,
              num_peaks_max=None,
              as_marker=True,
              connectivity=None,
              **kwargs):
        """
        Parameters
        ----------
        data : array
            image to be analyzed
        mask : array
            consider only regions where `mask == True`
        as_marker : bool, default=True
            if True, convert peaks location to labels array
        num_peaks_max : int, optional
        connectivity : int
            connetivity metric, used only if `as_marker==True`
        kwargs : dict
            extra arguments to `peak_local_max_adaptive`.  These overide self.peaks_kws
        Returns
        -------
        out :
            - if `as_marker`, then return label ar
            - else, return indicies of peaks
        Notes
        -----
        All of thes argmuents are in addition to self.peak_kws
        """

        kwargs = dict(self.peak_kws, **kwargs)
        if mask is not None:
            kwargs['mask'] = mask
        if num_peaks_max is not None:
            kwargs['num_peaks_max'] = num_peaks_max
        out = peak_local_max_adaptive(data, **kwargs)
        # combine markers
        if as_marker:
            out = morphology.label(out, connectivity=connectivity)
        return out

    def watershed(self, data, markers, mask, connectivity=None, **kwargs):
        """
        Parameters
        ----------
        data : image array
        markers : int or array of its with shape data.shape
        mask : array of bools of shape data.shape, optional
            if passed, mask==True indicates values to include
        connectivity : int
            connectivity to use in watershed
        kwargs : extra arguments to skimage.morphology.watershed
        Returns
        -------
        labels : array of ints
            Values > 0 correspond to found regions

        """

        if connectivity is None:
            connectivity = data.ndim
        kwargs = dict(self.watershed_kws, connectivity=connectivity, *kwargs)
        return morphology.watershed(data, markers=markers, mask=mask, **kwargs)

    def segment_lnpi(self,
                     lnpi,
                     find_peaks=True,
                     num_peaks_max=None,
                     connectivity=None,
                     peaks_kws=None,
                     watershed_kws=None):

        if find_peaks:
            if peaks_kws is None:
                peaks_kws = {}
            markers = self.peaks(lnpi.data,
                                 mask=~lnpi.mask,
                                 num_peaks_max=num_peaks_max,
                                 connectivity=connectivity,
                                 **peaks_kws)
        else:
            markers = num_peaks_max

        if watershed_kws is None:
            watershed_kws = {}
        labels = self.watershed(-lnpi.data,
                                markers=markers,
                                mask=~lnpi.mask,
                                connectivity=connectivity)
        return labels


class FreeEnergylnPi(object):
    """
    find/merge the transition energy between minima and barriers
    in lnPi

    here we define the free energy w = betaW = -ln(Pi)

    NOTE : this class used the image convension that
    mask == True indicates that the region includes the feature.
    This is oposite the masked array convension, where mask==True implies that region is masked out.
    """
    def __init__(self,
                 data,
                 masks,
                 convention='image',
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

        if connectivity is None:
            connectivity = data.ndim
        self._connectivity = connectivity

    @classmethod
    def from_labels(cls,
                    data,
                    labels,
                    connectivity=None,
                    features=None,
                    include_boundary=False,
                    **kwargs):
        """
        create FreeEnergylnPi from labels
        """
        masks, features = labels_to_masks(labels,
                                          features=features,
                                          convention='image',
                                          include_boundary=include_boundary,
                                          **kwargs)
        return cls(data=data, masks=masks, connectivity=connectivity)

    def _find_boundaries(self, idx):
        return segmentation.find_boundaries(self._masks[idx],
                                            connectivity=self._connectivity,
                                            mode='thick')

    #@gcached() # no need to cache
    @property
    def _boundaries(self):
        """boundary of each label"""
        return [self._find_boundaries(i) for i in self._index]

    #@gcached()
    @property
    def _boundaries_overlap(self):
        """overlap of boundaries"""
        boundaries = {}
        for i, j in itertools.combinations(self._index, 2):
            # instead of using foreground, maker sure that the boundary
            # is contained in eigher region[i] or region[j]
            overlap = (
                # overlap of boundary
                (self._boundaries[i] & self._boundaries[j])
                # overlap with
                &
                # with union of regions
                (self._masks[i] | self._masks[j]))

            if overlap.sum() == 0:
                overlap = None
            boundaries[i, j] = overlap
        return boundaries

    @property
    def _w_min(self):
        return -np.array([self._data[msk].max() for msk in self._masks])

    @gcached()
    def w_min(self):
        return self._w_min.reshape(-1, 1)

    @gcached()
    def w_tran(self, **kwargs):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        out = np.full((self._nfeature, ) * 2,
                      dtype=np.float,
                      fill_value=np.inf)

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
            nfeature_max = self._nfeature

        w_tran = self.w_tran.copy()
        w_min = self.w_min.copy()

        # keep track of keep/kill
        #mapping[keep] = [keep, merge_in_1, ...]
        #mapping = {i : [i] for i in self._index}
        mapping = {i: msk for i, msk in enumerate(self._masks)}
        for cnt in range(self._nfeature):
            # number of finite minima
            nfeature = len(mapping)
            #nfeature = np.isfinite(w_min).sum()

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

        # get masks
        masks = [mapping[i] for i in idx_min]

        # optionally convert image
        masks = masks_change_convention(masks, True, convention)

        return masks, w_tran, w_min


class wlnPi(FreeEnergylnPi):
    def __init__(self, phases):
        self._phases = phases
        base = self._phases[0]
        masks = [x.mask for x in self._phases]
        super(wlnPi, self).__init__(data=base.data,
                                    masks=masks,
                                    convention=False)

    @gcached()
    def delta_w(self):
        """wrap delta_w in xarray"""

        delta_w = self.w_tran - self.w_min

        xge = self._phases[0].xge

        dim_phase = self._phases._concat_dim
        dims = [dim_phase, dim_phase + '_nebr']

        coords = dict(zip(dims, [self._phases.index.values] * 2))
        coords = dict(xge.coords_state, **coords)
        return xr.DataArray(delta_w, dims=dims, coords=coords)

    def get_delta_w(self, idx, idx_nebr=None):
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
        dw : float
            - if only phase idx exists, dw = np.inf
            - if idx does not exists, dw = 0.0 (no barrier between idx and anything else)
            - else min of transition for idx to idx_nebr
        """
        p = self._phases

        has_idx = idx in p.index
        if not has_idx:
            return 0.0
        if idx_nebr is None:
            nebrs = p.index.drop(idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            nebrs = [_x for _x in idx_nebr if _x in p.index]

        if len(nebrs) == 0:
            return np.inf
        dw = (p.wlnPi.delta_w.sel(phase=idx,
                                  phase_nebr=nebrs).min('phase_nebr').values)
        return dw


def _get_delta_w(index, w):
    return (pd.DataFrame(
        w.delta_w,
        index=index,
        columns=index.get_level_values('phase').rename('phase_nebr')).stack())


@CollectionlnPi.decorate_accessor('wlnPi')
class wlnPivec(object):
    def __init__(self, parent):
        self._parent = parent
        self._use_joblib = getattr(self._parent, '_use_joblib', False)

    @gcached()
    def _items(self):
        indexes = []
        ws = []

        # s = self._parent._series
        # for meta, phases in s.groupby(allbut(s.index.names, 'phase')):
        for meta, phases in self._parent.groupby_allbut('phase'):
            indexes.append(phases.index)
            masks = [x.mask for x in phases.values]
            ws.append(
                FreeEnergylnPi(data=phases.iloc[0].data,
                               masks=masks,
                               convention=False))
        return indexes, ws

    @property
    def _indexes(self):
        return self._items[0]

    @property
    def _ws(self):
        return self._items[1]

    @gcached()
    def dw(self):
        """Series representation of delta_w"""
        seq = get_tqdm(zip(self._indexes, self._ws),
                       total=len(self._ws),
                       desc='wlnPi')
        out = parallel_map_func_starargs(_get_delta_w,
                                         items=seq,
                                         use_joblib=self._use_joblib,
                                         total=len(self._ws))
        out = pd.concat(out).rename('delta_w')
        return out

    @property
    def dwx(self):
        """xarray representation of delta_w"""
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
        dw : float
            - if only phase idx exists, dw = np.inf
            - if idx does not exists, dw = 0.0 (no barrier between idx and anything else)
            - else min of transition for idx to idx_nebr
        """

        delta_w = self.dwx
        #stack = {'sample': list(set(delta_w.coords) - {'phase', 'phase_nebr'})}
        #delta_w = self.dwx.stack(stack)

        #reindex so that has idx in phase
        reindex = delta_w.indexes['phase'].union(pd.Index([idx], name='phase'))
        delta_w = delta_w.reindex(phase=reindex, phase_nebr=reindex)

        # much simpler
        if idx_nebr is None:
            delta_w = delta_w.sel(phase=idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            if idx not in idx_nebr:
                idx_nebr.append(idx)
            nebrs = delta_w.indexes['phase_nebr'].intersection(nebrs)
            delta_w.sel(phase=idx, phase_nebr=nebrs)

        out = delta_w.min('phase_nebr').fillna(0.0)
        return out

        # has_idx = np.isinf(delta_w.sel(phase=idx, phase_nebr=idx))
        # # nebrs
        # index_nebr = delta_w.indexes['phase_nebr']
        # if idx_nebr is None:
        #     nebrs = index_nebr.drop(idx)
        # else:
        #     if not isinstance(idx_nebr, list):
        #         idx_nebr = [idx_nebr]
        #     nebrs = [_x for _x in idx_nebr if _x in index_nebr and _x != idx]

        # if len(nebrs) == 0:
        #     out = (delta_w.sel(phase=idx, phase_nebr=idx) * np.nan).fillna(np.inf)
        # else:
        #     out = delta_w.sel(phase=idx, phase_nebr=nebrs).min('phase_nebr')

        # out.loc[has_idx & out.isnull()] = np.inf
        # out.loc[~has_idx] = 0.0
        # return out.unstack('sample')

    def get_dw(self, idx, idx_nebr=None):
        return self.get_dwx(idx, idx_nebr).to_series()


@CollectionlnPi.decorate_accessor('wlnPi_single')
class wlnPi_single(wlnPivec):
    """
    stripped down version for single phase grouping
    """
    @gcached()
    def dwx(self):
        index = list(self._parent.index.get_level_values('phase'))
        masks = [x.mask for x in self._parent]
        w = FreeEnergylnPi(data=self._parent.iloc[0].data,
                           masks=masks,
                           convention=False)

        dw = w.w_tran - w.w_min
        dims = ['phase', 'phase_nebr']
        coords = dict(zip(dims, [index] * 2))
        return xr.DataArray(dw, dims=dims, coords=coords)

    @gcached()
    def dw(self):
        """Series representation of delta_w"""
        return self.dwx.to_series()

    def get_dw(self, idx, idx_nebr=None):
        dw = self.dwx
        index = dw.indexes['phase']

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
        return dw.sel(phase=idx, phase_nebr=nebrs).min('phase_nebr').values


class PhaseCreator(object):
    """
    Helper class to create phases
    """
    def __init__(self,
                 nmax,
                 nmax_peak=None,
                 ref=None,
                 segmenter=None,
                 segment_kws=None,
                 tag_phases=None,
                 phases_factory=CollectionlnPi.from_list,
                 FreeEnergylnPi_kws=None,
                 merge_kws=None):
        """
        Parameters
        ----------
        nmax : int
            number of phases to construct
        nmax_peak : int, optional
            if specified, the allowable number of peaks to locate.
            This can be useful for some cases.  These phases will be merged out at the end.
        ref : MaskedlnPi object, optional
        segmenter : Segmenter object, optional
            segmenter object to create labels/masks
        Freeenergy_kws : dict, optional
            dictionary of parameters for the creation of a FreeenergylnPi object
        """

        if nmax_peak is None:
            nmax_peak = nmax * 2
        self.nmax = nmax
        self.ref = ref

        if segmenter is None:
            segmenter = Segmenter()
        self.segmenter = segmenter

        self.tag_phases = tag_phases

        self.phases_factory = phases_factory

        if segment_kws is None:
            segment_kws = {}
        self.segment_kws = segment_kws
        self.segment_kws['num_peaks_max'] = nmax_peak

        if FreeEnergylnPi_kws is None:
            FreeEnergylnPi_kws = {}
        self.FreeEnergylnPi_kws = FreeEnergylnPi_kws

        if merge_kws is None:
            merge_kws = {}
        merge_kws = dict(merge_kws, convention=False, nfeature_max=self.nmax)
        self.merge_kws = merge_kws

    def _merge_phase_ids(sel, ref, phase_ids, lnpis):
        """
        perform merge of phase_ids/index
        """
        from scipy.spatial.distance import pdist

        if len(phase_ids) == 1:
            # only single phase_id
            return phase_ids, lnpis

        phase_ids = np.array(phase_ids)
        dist = pdist(phase_ids.reshape(-1, 1)).astype(np.int)
        if not np.any(dist == 0):
            # all different
            return phase_ids, lnpis

        phase_ids_new = []
        masks_new = []
        for idx in np.unique(phase_ids):
            where = np.where(idx == phase_ids)[0]
            mask = np.all([lnpis[i].mask for i in where], axis=0)

            phase_ids_new.append(idx)
            masks_new.append(mask)
        lnpis_new = ref.list_from_masks(masks_new, convention=False)

        return phase_ids_new, lnpis_new

    def build_phases(self,
                     lnz=None,
                     ref=None,
                     efac=None,
                     nmax=None,
                     nmax_peak=None,
                     connectivity=None,
                     reweight_kws=None,
                     merge_phase_ids=True,
                     phases_factory=None,
                     phase_kws=None):
        """
        build phase
        """

        if ref is None:
            if self.ref is None:
                raise ValueError('must specify ref or self.ref')
            ref = self.ref

        # reweight
        if lnz is not None:
            if reweight_kws is None:
                reweight_kws = {}
            ref = ref.reweight(lnz, **reweight_kws)

        if nmax is None:
            nmax = self.nmax

        if nmax_peak is None:
            nmax_peak = nmax * 2

        if nmax > 1:
            # labels
            kws = dict(self.segment_kws, num_peaks_max=nmax_peak)
            if connectivity is not None:
                kws['connectivity'] = connectivity
            labels = self.segmenter.segment_lnpi(lnpi=ref, **kws)

            # wlnpi
            kws = dict(self.FreeEnergylnPi_kws)
            if connectivity is not None:
                kws['connectivity'] = connectivity
            wlnpi = FreeEnergylnPi.from_labels(data=ref.data,
                                               labels=labels,
                                               **kws)

            # merge
            kws = dict(self.merge_kws, nfeature_max=nmax)
            if efac is not None:
                kws['efac'] = efac
            masks, wtran, wmin = wlnpi.merge_regions(**kws)

            # list of lnpi
            lnpis = ref.list_from_masks(masks, convention=False)

            # tag phases?
            if self.tag_phases is not None:
                index = self.tag_phases(lnpis)
                if merge_phase_ids:
                    index, lnpis = self._merge_phase_ids(ref, index, lnpis)

            else:
                index = list(range(len(lnpis)))
        else:
            lnpis = [ref]
            index = [0]

        if phases_factory is None:
            phases_factory = self.phases_factory
        if isinstance(phases_factory,
                      str) and phases_factory.lower() == 'none':
            phases_factory = None
        if phases_factory is not None:
            if phase_kws is None:
                phase_kws = {}
            return phases_factory(items=lnpis, index=index, **phase_kws)
        else:
            return lnpis, index

    def build_phases_mu(self, lnz):
        return BuildPhases_mu(lnz, self)

    def build_phases_dmu(self, dlnz):
        return BuildPhases_dmu(dlnz, self)


class _BuildPhases(object):
    """
    class to build phases object from scalar mu's
    """
    def __init__(self, X, phase_creator):

        self._phase_creator = phase_creator
        self.X = X

    @property
    def X(self):
        return self._X

    @property
    def phase_creator(self):
        return self._phase_creator

    @X.setter
    def X(self, X):
        assert sum([x is None for x in X]) == 1
        self._X = X
        self._ncomp = len(self._X)
        self._index = self._X.index(None)
        self._set_params()

    @property
    def index(self):
        return self._index

    def _set_params(self):
        pass

    def _get_lnz(self, lnz_index):
        # to be implemented in child class
        raise NotImplementedError

    def __call__(self, lnz_index, *args, **kwargs):
        lnz = self._get_lnz(lnz_index)
        return self._phase_creator.build_phases(lnz=lnz, *args, **kwargs)


# from .utils import get_lnz_iter
class BuildPhases_mu(_BuildPhases):
    def __init__(self, lnz, phase_creator):
        """
        Parameters
        ----------
        lnz : list
            list with one element equal to None.  This is the component which will be varied
            For example, lnz=[lnz0,None,lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
            vary component 1
        phase_creator : PhaseCreator object
        """
        super().__init__(X=lnz, phase_creator=phase_creator)

    def _get_lnz(self, lnz_index):
        lnz = self.X.copy()
        lnz[self.index] = lnz_index
        return lnz


class BuildPhases_dmu(_BuildPhases):
    def __init__(self, dlnz, phase_creator):
        """
        Parameters
        ----------
        dlnz : list
            list with one element equal to None.  This is the component which will be varied
            For example, dlnz=[dlnz0,None,dlnz2] implies use values of dlnz0,dlnz2 for components 0 and 2, and
            vary component 1
            dlnz_i = lnz_i - lnz_index, where lnz_index is the value varied.
        phase_creator : PhaseCreator object
        """
        super().__init__(X=dlnz, phase_creator=phase_creator)

    def _set_params(self):
        self._dlnz = np.array([x if x is not None else 0.0 for x in self.X])

    def _get_lnz(self, lnz_index):
        return self._dlnz + lnz_index


from functools import lru_cache


@lru_cache(maxsize=10)
def get_default_PhaseCreator(nmax):
    return PhaseCreator(nmax=nmax)
