"""
Routines to segment lnPi

 1. find max/peaks in lnPi
 2. segment lnPi about these peaks
 3. determine free energy difference between segments
    a. Merge based on low free energy difference
 4. combination of 1-3.
"""

import warnings
from collections.abc import Iterable
from functools import lru_cache

import bottleneck
import numpy as np
from skimage import feature, morphology, segmentation

from ._docstrings import _prepare_shared_docs, _shared_docs, docfiller
from .lnpienergy import wFreeEnergy
from .lnpiseries import lnPiCollection

# * Common doc strings

_shared_docs_local = {
    "data": """
    data : array-like
        Image data to analyze
    """,
    "min_distance": """
    min_distance : int or sequence of ints, default=(5, 10, 15, 20, 25)
        min_distance parameter.  If sequence, then call
        :func:`~skimage.feature.peak_local_max` until number of peaks ``<=num_peaks_max``.
    """,
    "connectivity_morphology": """
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
        Accepted values are ranging from 1 to ``input.ndim``. If ``None``, a full connectivity of ``input.ndim`` is used.
    """,
    "connectivity_watershed": """
    connectivity : ndarray, optional
        An array with the same number of dimensions as image whose non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of the dimension of the image.
    """,
    "num_peaks_max": """
    num_peaks_max : int, optional.
        Max number of maxima/peaks to find. If not specified, any number of peaks allowed.
    """,
    "peak_style": """
    style : {'indices', 'mask', 'marker'}
        Controls output style

        * indices : indices of peaks
        * mask : bool array marking peak locations
        * marker : array of ints

    """,
    "markers": """
    markers : int, or ndarray of int, same shape as image, optional
        The desired number of markers, or an array marking the basins with the values to be assigned in the label matrix.
        Zero means not a marker. If None (no markers given), the local minima of the image are used as markers.
    """,
}

_shared_docs_local = _prepare_shared_docs(_shared_docs_local)
docfiller_shared = docfiller(**(dict(_shared_docs, **_shared_docs_local)))

# docfiller_shared = docfiller(**dict(_shared_docs, **_prepare_shared_docs(_shared_docs_local)))


@docfiller_shared
def peak_local_max_adaptive(
    data,
    mask=None,
    min_distance=None,
    threshold_rel=0.00,
    threshold_abs=0.2,
    num_peaks_max=None,
    style="indices",
    connectivity=None,
    errors="warn",
    **kwargs
):
    """
    Find local max with fall backs min_distance and filter.

    This is an adaptation of :func:`~skimage.feature.peak_local_max`, which is
    called iteratively until the number of `peaks` is less than `num_peaks_max`.

    Parameters
    ----------
    {data}
    {mask_image}
    {min_distance}
    threshold_rel, threshold_abs : float
        thresholds parameters
    {num_peaks_max}
    {peak_style}
    {connectivity_morphology}
    indices : bool, optional, default=True
        if True, return indicies of peaks.
        if False, return array of ints of shape `data.shape` with peaks
        marked by value > 0.
    errors : {{'ignore','raise','warn'}}, default='warn'
        - If raise, raise exception if npeaks > num_peaks_max
        - If ignore, return all found maxima
        - If warn, raise warning if npeaks > num_peaks_max
    **kwargs
        Extra arguments to :func:`~skimage.feature.peak_local_max`

    Returns
    -------
    out : indices or mask
        Depending on the value of `indices`.

    Notes
    -----
    The option `mask` is passed as the value `labels` in :func:`~skimage.feature.peak_local_max`

    See Also
    --------
    ~skimage.feature.peak_local_max
    ~skimage.morphology.label
    """

    assert style in ["indices", "mask", "marker"]

    if min_distance is None:
        min_distance = [5, 10, 15, 20, 25]

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)
    kwargs = dict(dict(exclude_border=False), **kwargs)

    n = idx = None
    for md in min_distance:
        idx = feature.peak_local_max(
            data,
            min_distance=md,
            labels=mask,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            # this option removed in future
            **kwargs
        )

        n = len(idx)
        if n <= num_peaks_max:
            break

    if n is None or idx is None:
        raise ValueError("failed to find peaks")

    if n > num_peaks_max:
        if errors == "ignore":
            pass
        elif errors in ("raise", "ignore"):
            message = "{} maxima found greater than {}".format(n, num_peaks_max)
            if errors == "raise":
                raise RuntimeError(message)
            else:
                warnings.warn(message)

    idx = tuple(idx.T)
    if style == "indices":
        out = idx
    else:
        out = np.zeros_like(data, dtype=bool)
        out[idx] = True

        if style == "marker":
            out = morphology.label(out, connectivity=connectivity)

    return out


@docfiller_shared
class Segmenter(object):
    """
    Data segmenter:


    Parameters
    ----------
    {min_distance}
    {peak_kws}
    {watershed_kws}
    """

    def __init__(self, min_distance=None, peak_kws=None, watershed_kws=None):

        if min_distance is None:
            min_distance = [1, 5, 10, 15, 20]

        if peak_kws is None:
            peak_kws = {}
        self.peak_kws = peak_kws

        if watershed_kws is None:
            watershed_kws = {}
        self.watershed_kws = watershed_kws

    @docfiller_shared
    def peaks(self, data, mask=None, num_peaks_max=None, style="marker", **kwargs):
        """
        Interface to :func:`peak_local_max_adaptive` with default values from `self`.


        Parameters
        ----------
        {data}
        {num_peaks_max}
        {peak_style}
        {mask_image}

        **kwargs
            Extra arguments to :func:`peak_local_max_adaptive`.

        Returns
        -------
        out :
            If ``style=='marker'``, then return label array.  Otherwise,
            return indicies of peaks.

        Notes
        -----
        Any value not set will be inherited from `self.peak_kws`


        See Also
        --------
        peak_local_max_adaptive

        """

        if mask is not None:
            kwargs["mask"] = mask
        if num_peaks_max is not None:
            kwargs["num_peaks_max"] = num_peaks_max
        kwargs["style"] = style
        kwargs = dict(self.peak_kws, **kwargs)
        out = peak_local_max_adaptive(data, **kwargs)

        return out

    @docfiller_shared
    def watershed(self, data, markers, mask, connectivity=None, **kwargs):
        """
        Interface to :func:`skimage.segmentation.watershed` function

        Parameters
        ----------
        {data}
        {markers}

        {mask_image}
        connectivity : int
            connectivity to use in watershed
        {connectivity_watershed}
        **kwargs
            Extra arguments to :func:`~skimage.segmentation.watershed`

        Returns
        -------
        labels : array of ints
            Values > 0 correspond to found regions

        See Also
        --------
        ~skimage.segmentation.watershed
        """

        if connectivity is None:
            connectivity = data.ndim
        kwargs = dict(self.watershed_kws, connectivity=connectivity, *kwargs)
        return segmentation.watershed(data, markers=markers, mask=mask, **kwargs)

    @docfiller_shared
    def segment_lnpi(
        self,
        lnpi,
        markers=None,
        find_peaks=True,
        num_peaks_max=None,
        connectivity=None,
        peaks_kws=None,
        watershed_kws=None,
    ):
        """
        Perform segmentations of lnPi object using watershed on negative of lnPi data.

        Parameters
        ----------
        lnPi : lnPiMasked
            Object to be segmented
        {markers}

        find_peaks : bool, default=True
            If True, use :func:`peak_local_max_adaptive` to construct `markers`.
        {num_peaks_max}
        {connectivity_watershed}
        {peak_kws}
        {watershed_kws}

        Returns
        -------
        {labels}


        See Also
        --------
        Segmenter.watershed
        skimage.morphology.watershed

        """

        if markers is None:
            if find_peaks:
                if peaks_kws is None:
                    peaks_kws = {}
                markers = self.peaks(
                    lnpi.data,
                    mask=~lnpi.mask,
                    num_peaks_max=num_peaks_max,
                    connectivity=connectivity,
                    **peaks_kws
                )
            else:
                markers = num_peaks_max

        if watershed_kws is None:
            watershed_kws = {}
        labels = self.watershed(
            -lnpi.data, markers=markers, mask=~lnpi.mask, connectivity=connectivity
        )
        return labels


class PhaseCreator(object):
    """
    Helper class to create phases

    Parameters
    ----------
    nmax : int
        Maximum number of phases to allow
    nmax_peak : int, optional
        if specified, the allowable number of peaks to locate.
        This can be useful for some cases.  These phases will be merged out at the end.
    ref : lnPiMasked, optional
    segmenter : Segmenter object, optional
        segmenter object to create labels/masks. Defaults to using base segmenter.
    segment_kws : mapping, optional
        Optional arguments to be passed to :meth`Segmenter.segmenter_lnpi`.
    tag_phases : callable, optional
        Optional funciton which takes a list of :class:`lnpy.lnPiMasked` objects
        and returns on integer label for each object.
    phases_factory : callable, optional
        Factory function for returning Collection from a list of :class:`lnpy.lnPiMasked` object.
        Defaults to :meth:`lnpy.lnPiCollection.from_list`.
    lnPiFreeEnergy_kws : mapping, optional
        Optional arguments to ...
    merge_kws : mapping, optional
        Optional arguments to ...

    """

    def __init__(
        self,
        nmax,
        nmax_peak=None,
        ref=None,
        segmenter=None,
        segment_kws=None,
        tag_phases=None,
        phases_factory=None,
        lnPiFreeEnergy_kws=None,
        merge_kws=None,
    ):

        if phases_factory is None:
            phases_factory = lnPiCollection.from_list

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
        self.segment_kws["num_peaks_max"] = nmax_peak

        if lnPiFreeEnergy_kws is None:
            lnPiFreeEnergy_kws = {}
        self.lnPiFreeEnergy_kws = lnPiFreeEnergy_kws

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
        dist = pdist(phase_ids.reshape(-1, 1)).astype(int)
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

    def build_phases(
        self,
        lnz=None,
        ref=None,
        efac=None,
        nmax=None,
        nmax_peak=None,
        connectivity=None,
        reweight_kws=None,
        merge_phase_ids=True,
        merge_phases=True,
        phases_factory=None,
        phase_kws=None,
        segment_kws=None,
        lnPiFreeEnergy_kws=None,
        merge_kws=None,
        tag_phases=None,
    ):
        """
        Construct 'phases' for a lnPi object.

        This is quite an involved process.  The steps are

        * Optionally find the location of the maxima in lnPi.
        * Segment lnpi using watershed
        * Merge phases which are energentically similar
        * Optionally merge phases which have the same `phase_id`

        Parameters
        ----------
        lnz : int or sequence of ints, optional
            lnz value to evaluate `ref` at.  If not specified, use
            `ref.lnz`
        ref : lnPiMasked object
            Object to be segmented
        efac : float, optional
            Optional value to use in energetic merging of phases.
        nmax : int, optional
            Maximum number of phases.  Defaults to `self.nmax`
        nmax_peak : int, optional
            Maximum number of peaks to allow in :func:`peak_local_max_adaptive`.
            Note that this value can be larger than `nmax`.  Defaults to `self.nmax_peak`.
        {connectivity_morphology}
        reweight_kws : mapping, optional
            Extra arguments to `ref.reweight`
        merge_phase_ids : bool, default = True
            If True and calling `tag_phases` routine, merge phases with same phase_id.
        phases_factory : callable, optional
            Function to convert list of phases into Phases object.
            Defaults to `self.phases_factory`.
        phase_kws : mapping, optional
            Extra arguments to `phases_factory`
        segment_kws : mapping, optional
            Extra arguments to `self.segmenter.segment_lnpi`
        lnPiFreeEnergy_kws : mapping, optional
            Extra arguments to free energy calculation
        merge_kws : mapping, optional
            Extra arguments to merge
        tag_phases : callable, optional
            Funciton to tag phases.  Defaults to `self.tag_phases`

        """

        def _combine_kws(class_kws, passed_kws, **default_kws):
            if class_kws is None:
                class_kws = {}
            if passed_kws is None:
                passed_kws = {}
            return dict(class_kws, **default_kws, **passed_kws)

        if ref is None:
            if self.ref is None:
                raise ValueError("must specify ref or self.ref")
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

        connectivity_kws = {}
        if connectivity is not None:
            connectivity_kws["connectivity"] = connectivity

        if nmax > 1:
            # segment lnpi using watershed
            segment_kws = _combine_kws(
                self.segment_kws,
                segment_kws,
                num_peaks_max=nmax_peak,
                **connectivity_kws
            )
            labels = self.segmenter.segment_lnpi(lnpi=ref, **segment_kws)

            # analyze w = - lnPi
            lnPiFreeEnergy_kws = _combine_kws(
                self.lnPiFreeEnergy_kws, lnPiFreeEnergy_kws, **connectivity_kws
            )
            wlnpi = wFreeEnergy.from_labels(
                data=ref.data, labels=labels, **lnPiFreeEnergy_kws
            )

            if merge_phases:
                # merge
                other_kws = {} if efac is None else {"efac": efac}
                merge_kws = _combine_kws(
                    self.merge_kws, merge_kws, nfeature_max=nmax, **other_kws
                )
                masks, _, _ = wlnpi.merge_regions(**merge_kws)
            else:
                masks = wlnpi.masks

            # list of lnpi
            lnpis = ref.list_from_masks(masks, convention=False)

            # tag phases?
            if tag_phases is None:
                tag_phases = self.tag_phases
            if tag_phases is not None:
                index = tag_phases(lnpis)
                if merge_phase_ids:
                    index, lnpis = self._merge_phase_ids(ref, index, lnpis)
            else:
                index = list(range(len(lnpis)))
        else:
            lnpis = [ref]
            index = [0]

        if phases_factory is None:
            phases_factory = self.phases_factory
        if isinstance(phases_factory, str) and phases_factory.lower() == "none":
            phases_factory = None
        if phases_factory is not None:
            if phase_kws is None:
                phase_kws = {}
            return phases_factory(items=lnpis, index=index, **phase_kws)
        else:
            return lnpis, index

    def build_phases_mu(self, lnz):
        """
        Create constructor at fixed values of `mu`

        See Also
        --------
        BuildPhases_mu
        """
        return BuildPhases_mu(lnz, self)

    def build_phases_dmu(self, dlnz):
        """
        Create constructor at fixed values of `dmu`.

        See Also
        --------
        BuildPhases_dmu

        """
        return BuildPhases_dmu(dlnz, self)


class BuildPhasesBase(object):
    """
    Base class to build Phases objecs from scalar values of `lnz`.
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
        """
        Build phases from scalar value of lnz.
        """
        lnz = self._get_lnz(lnz_index)
        return self._phase_creator.build_phases(lnz=lnz, *args, **kwargs)


# from .utils import get_lnz_iter
class BuildPhases_mu(BuildPhasesBase):
    """
    create phases from scalar value of mu for fixed value of mu for other species

    Parameters
    ----------
    lnz : list
        list with one element equal to None.  This is the component which will be varied
        For example, lnz=[lnz0, None, lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
        vary component 1
    phase_creator : PhaseCreator object
    """

    def __init__(self, lnz, phase_creator):
        super().__init__(X=lnz, phase_creator=phase_creator)

    def _get_lnz(self, lnz_index):
        lnz = self.X.copy()
        lnz[self.index] = lnz_index
        return lnz


class BuildPhases_dmu(BuildPhasesBase):
    """
    Create phases from scalar value of mu at fixed value of dmu for other species

    Parameters
    ----------
    dlnz : list
        list with one element equal to None.  This is the component which will be varied
        For example, dlnz=[dlnz0,None,dlnz2] implies use values of dlnz0,dlnz2
        for components 0 and 2, and vary component 1.
        dlnz_i = lnz_i - lnz_index, where lnz_index is the value varied.
    phase_creator : :class:`lnpy.PhaseCreator`
    """

    def __init__(self, dlnz, phase_creator):
        super().__init__(X=dlnz, phase_creator=phase_creator)

    def _set_params(self):
        self._dlnz = np.array([x if x is not None else 0.0 for x in self.X])

    def _get_lnz(self, lnz_index):
        return self._dlnz + lnz_index


@lru_cache(maxsize=10)
def get_default_PhaseCreator(nmax):
    return PhaseCreator(nmax=nmax)
