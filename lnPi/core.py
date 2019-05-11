import itertools
from collections import Iterable

import numpy as np
import xarray as xr
import pandas as pd

from scipy.spatial.distance import cdist, pdist, squareform

from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

from functools import wraps, partial

from scipy.ndimage import filters
from lnPi.cached_decorators import cached_clear, cached, cached_func, gcached
from lnPi._utils import _interp_matrix, get_mu_iter
from lnPi._segment import (_indices_to_markers, _labels_watershed,
                           labels_to_masks, masks_to_labels)

from lnPi.spinodal import get_spinodal
from lnPi.binodal import get_binodal_point
from lnPi.molfrac import find_mu_molfrac
from functools import wraps, partial



# cached indices
_INDICES = {}
def _get_indices(shape):
    if shape not in _INDICES:
        _INDICES[shape] = np.indices(shape)
    return _INDICES[shape]



class lnPi(np.ma.MaskedArray):
    """
    class to store masked ln[Pi(n0,n1,...)].
    shape is (N0,N1,...) where Ni is the span of each dimension)

    Attributes
    ----------
    self : masked array containing lnPi

    mu : chemical potential for each component

    coords : coordinate array (ndim,N0,N1,...)

    pi : exp(lnPi)

    pi_norm : pi/pi.sum()

    nave : average number of particles of each component
    cax = divider.append_axes("right",
                              size="2%",
                              pad=cbar_pad)


    molfrac : mol fraction of each component

    Omega : Omega system, relative to lnPi[0,0,...,0]


    set_data : set data array

    set_mask : set mask array


    argmax_local : local argmax (in np.where output form)

    get_phases : return phases


    zeromax : set lnPi = lnPi - lnPi.max()

    pad : fill in masked points by interpolation

    adjust : zeromax and/or pad


    reweight : create new lnPi at new mu

    new_mask : new object (default share data) with new mask

    add_mask : create object with mask = self.mask + mask

    smooth : create smoothed object


    from_file : create lnPi object from file

    from_data : create lnPi object from data array
    """

    def __new__(cls,
                data,
                mu=None,
                zeromax=True,
                pad=False,
                num_phases_max=None,
                volume=None,
                beta=None,
                **kwargs):
        """
        constructor

        Parameters
        ----------
        data : array-like
         data for lnPi

        mu : array-like (Default None)
         if None, set mu=np.zeros(data.ndim)

        zeromax : bool (Default False)
         if True, shift lnPi = lnPi - lnPi.max()


        pad : bool (Default False)
         if True, pad masked region by interpolation


        **kwargs : arguments to np.ma.array
         e.g., mask=...

        """

        obj = np.ma.array(data, **kwargs).view(cls)

        fv = kwargs.get('fill_value', None) or getattr(data, 'fill_value',
                                                       None)
        if fv is not None:
            obj.set_fill_value(fv)

        obj._optinfo.update(
            mu=mu, num_phases_max=num_phases_max, volume=volume, beta=beta)
        obj.adjust(zeromax=zeromax, pad=pad, inplace=True)

        return obj

    ##################################################
    #caching
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._clear_cache()

    def _clear_cache(self):
        self._cache = {}

    ##################################################
    #properties
    @property
    def optinfo(self):
        return self._optinfo

    @property
    def mu(self):
        mu = self._optinfo.get('mu', None)
        if mu is None:
            mu = np.zeros(self.ndim)
        mu = np.atleast_1d(mu).astype(self.dtype)
        if len(mu) != self.ndim:
            raise ValueError('bad len on mu %s' % mu)
        return mu

    @property
    def num_phases_max(self):
        return self._optinfo.get('num_phases_max', None)

    @property
    def volume(self):
        return self._optinfo.get('volume', None)

    @property
    def beta(self):
        return self._optinfo.get('beta', None)

    def __setitem__(self, index, value):
        self._clear_cache()
        super().__setitem__(index, value)

    @gcached()
    def dims_n(self):
        return ['n_{}'.format(i) for i in range(self.ndim)]

    @gcached()
    def dims_mu(self):
        return ['mu_{}'.format(i) for i in range(self.ndim)]

    @gcached()
    def dims_comp(self):
        return ['component']

    @gcached()
    def coords_state(self):
        coords = dict(zip(self.dims_mu, self.mu))
        coords['beta'] = self.beta
        coords['volume'] = self.volume
        return coords

    @gcached()
    def attrs(self):
        return {
            'dims_n': self.dims_n,
            'dims_mu': self.dims_mu,
            'dims_comp': self.dims_comp,
            'dims_state': list(self.coords_state.keys()),
            'lnpi_zero': self.lnpi_zero,
            'state_as_attrs': 0,
            'num_phases_max': self.num_phases_max
        }

    @gcached()
    def lnpi(self):
        return xr.DataArray(
            self.filled(np.nan),
            dims=self.dims_n,
            coords=self.coords_state,
            name='lnpi',
            attrs=self.attrs)

    @gcached()
    def ncoords(self):
        """particle number for each particle dimension"""
        return xr.DataArray(
            _get_indices(self.shape),
            dims=self.dims_comp + self.dims_n,
            coords=self.coords_state)

    @gcached()
    def chempot(self):
        return xr.DataArray(
            self.mu,
            dims=self.dims_comp,
            coords=self.coords_state,
            name='chempot')

    @gcached()
    def lnpi_max(self):
        return self.lnpi.max(self.dims_n).rename('lnpi_max')

    @gcached()
    def lnpi_zero(self):
        # kws = {k:0 for k in self.dims_n}
        # return self.lnpi.sel(**kws)
        # this was a nice idea, but then if masked out 0 point, then nan
        return self.data.reshape(-1)[0]

    #calculated properties
    @gcached()
    def pi(self):
        """
        basic pi = exp(lnpi)
        """
        return np.exp(self.lnpi - self.lnpi_max).rename('pi')

    @gcached()
    def pi_sum(self):
        return self.pi.sum(self.dims_n).rename('pi_sum')

    @gcached()
    def pi_norm(self):
        return (self.pi / self.pi_sum).rename('pi_norm')

    @gcached()
    def nave(self):
        """average number of particles of each component"""
        return (self.pi_norm * self.ncoords).sum(self.dims_n).rename('nave')

    @property
    def density(self):
        """density of each component"""
        return (self.nave / self.volume).rename("density")

    @gcached()
    def nvar(self):
        return (self.pi_norm * (self.ncoords - self.nave)**2).sum(
            self.dims_n).rename('nvar')

    @property
    def molfrac(self):
        n = self.nave
        return (n / n.sum()).rename('molfrac')

    @gcached(prop=False)
    def omega(self, zval=None):
        """
        get omega = zval - ln(sum(pi))

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """

        if zval is None:
            zval = self.lnpi_zero - self.lnpi_max
        return ((zval - np.log(self.pi_sum)) / self.beta).rename('omega')

    @gcached()
    def helmholtz_nvt(self):
        """
        Helmholtz free energy in canonical ensemble
        """
        return (-(self.lnpi - self.lnpi_zero) / self.beta +
                (self.ncoords * self.chempot).sum(self.dims_comp))

    @property
    def ncomp(self):
        return self.ndim

    ##################################################
    #setters
    @cached_clear()
    def set_data(self, val):
        self.data[:] = val

    @cached_clear()
    def set_mask(self, val):
        self.mask[:] = val

    ##################################################
    #adjusters
    def zeromax(self, inplace=False):
        """
        shift so that lnpi.max() == 0
        """

        if inplace:
            y = self
            self._clear_cache()
        else:
            y = self.copy()

        shift = self.max()

        y.set_data(y.data - shift)
        return y

    def pad(self, inplace=False):
        """
        pad self.data rows and columns by last value
        """
        if self.ndim == 1:
            msk = self.mask
            last = self[~msk][-1]

            data = self.data.copy()
            data[msk] = last

        elif self.ndim == 2:
            data = _interp_matrix(self.data, self.mask)
        else:
            raise ValueError('padding only implemented for ndim<=2')

        if inplace:
            y = self
            self._clear_cache()
        else:
            y = self.copy()
        y.set_data(data)
        return y

    def adjust(self, zeromax=False, pad=False, inplace=False):
        """
        do multiple adjustments in one go
        """

        if inplace:
            z = self
        else:
            z = self.copy()
        if zeromax:
            z.zeromax(inplace=True)
        if pad:
            z.pad(inplace=True)
        return z

    ##################################################
    #new object/modification
    def reweight(self, mu, zeromax=False, pad=False):
        """
        get lnpi at new mu

        Parameters
        ----------
        mu : array-like
            chem. pot. for new state point

        beta : float
            inverse temperature

        zeromax : bool (Default False)

        pad : bool (Default False)

        phases : dict


        Returns
        -------
        lnPi(mu)
        """

        Z = self.copy()
        Z._optinfo['mu'] = mu
        dmu = Z.mu - self.mu

        #s = _get_shift(self.shape,dmu)*self.beta
        #get shift
        #i.e., N * (mu_1 - mu_0)
        # note that this is (for some reason)
        # faster than doing the (more natural) options:
        # N = self.ncoords.values
        # shift = 0
        # for i, m in enumerate(dmu):
        #     shift += N[i,...] * m
        # or
        # shift = (self.ncoords.values.T * dmu).sum(-1).T

        shift = np.zeros([], dtype=float)
        for i, (s, m) in enumerate(zip(self.shape, dmu)):
            shift = np.add.outer(shift, np.arange(s) * m)

        #scale by beta
        shift *= self.beta

        Z.set_data(Z.data + shift)

        Z.adjust(zeromax=zeromax, pad=pad, inplace=True)

        return Z


    def new_mask(self, mask=None, **kwargs):
        """
        create copy with new mask
        """
        return self.__class__(
            self.data, mask=mask, **dict(self._optinfo, **kwargs))

    def add_mask(self, mask, **kwargs):
        """
        logical or of self.mask and mask

        Note, if want a copy, pass copy=True
        """
        return self.new_mask(mask=mask + self.mask, **kwargs)

    def smooth(self,
               sigma=4,
               mode='nearest',
               truncate=4,
               inplace=False,
               zeromax=False,
               pad=False,
               **kwargs):
        """
        apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool (Default False)
         if True, do inplace modification.


        **kwargs : (Default sigma=4, mode='nearest',truncate=4)
         arguments to filters.gaussian_filter
        """

        if inplace:
            Z = self
        else:
            Z = self.copy()

        Z._clear_cache()
        Z.adjust(zeromax=zeromax, pad=pad, inplace=True)
        filters.gaussian_filter(
            Z.data,
            output=Z.data,
            mode=mode,
            truncate=truncate,
            sigma=sigma,
            **kwargs)
        if not inplace:
            return Z

    def to_dataarray(self, mask=False, state_as_attrs=False, **kwargs):
        """
        create a xarray.DataArray from self.

        Parameters
        ----------
        mask : bool, default=False
            if True, include mask in output
        state_as_attrs: bool, default=False
            if True, demote `self.coords_state` to attributes
            This is useful if puttine self.to_dataarray() into `xr.Dataset`
        kwargs : extra arguments to xarray.DataArray
        """
        coords = {}
        attrs = self.attrs.copy()
        attrs['state_as_attrs'] = int(state_as_attrs)
        if state_as_attrs:
            attrs.update(**self.coords_state)
        else:
            coords.update(**self.coords_state)

        if mask:
            coords['mask'] = (self.dims_n, self.mask)
        return xr.DataArray(
            self.data,
            dims=self.dims_n,
            coords=coords,
            attrs=attrs,
            name='lnpi',
            **kwargs)

    @classmethod
    def from_dataarray(cls, da, state_as_attrs=None, **kwargs):
        """
        create a lnPi object from xarray.DataArray
        """

        kws = {}
        kws['data'] = da.values
        if 'mask' in da.coords:
            kws['mask'] = da.mask.values
        else:
            kws['mask'] = da.isnull().values

        # where are state variables
        if state_as_attrs is None:
            state_as_attrs = bool(da.attrs['state_as_attrs'])
        if state_as_attrs:
            # state variables from attrs
            c = da.attrs
        else:
            c = da.coords

        mu = []
        for k in da.attrs['dims_state']:
            if 'mu' in k:
                mu.append(c[k])
            else:
                kws[k] = c[k]
        kws['mu'] = mu

        # num_phases_max
        kws['num_phases_max'] = da.attrs['num_phases_max']

        # any overrides
        kwargs = dict(kws, **kwargs)

        return cls(**kwargs)

    ##################################################
    #maxima
    def _peak_local_max(self,
                        min_distance=1,
                        threshold_rel=0.00,
                        threshold_abs=0.2,
                        **kwargs):
        """
        find local maxima using skimage.feature.peak_local_max

        Parameters
        ----------
        min_distance : int (Default 1)
          min_distance parameter to peak_local_max

        **kwargs : arguments to peak_local_max (see docs)
         Defaults to min_distance=1,exclude_border=False

        Returns
        -------
        out : tupe of ndarrays
         indices of self where local max

        n : int
         number of maxima found
        """

        kwargs = dict(dict(exclude_border=False, labels=~self.mask), **kwargs)

        data = self.data - np.nanmin(self.data)

        x = peak_local_max(
            data,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            **kwargs)

        if kwargs.get('indices', True):
            #return indices
            x = tuple(x.T)
            n = len(x[0])
        else:
            n = x.sum()

        return x, n

    def argmax_local(self,
                     min_distance=[5, 10, 15, 20, 25],
                     threshold_rel=0.00,
                     threshold_abs=0.2,
                     smooth='fail',
                     smooth_kwargs={},
                     num_phases_max=None,
                     info=False,
                     **kwargs):
        """
        find local max with fall backs min_distance and filter

        Parameters
        ----------
        min_distance : int or iterable (Default 15)
            min_distance parameter to self.peak_local_max.
            if min_distance is iterable, if num_phase>num_phase_max, try next


        smooth : bool or str (Default 'fail')
            if True, use smooth only
            if False, no smooth
            if 'fail', do smoothing only if non-smooth fails

        smooth_kwargs : dict
            extra arguments to self.smooth

        num_phases_max : int (Default None)
            max number of maxima to find. if None, use self.num_phases_max


        info : bool (Default False)
            if True, return out_info

        **kwargs : extra arguments to peak_local_max

        Returns
        -------
        out : tuple of ndarrays
            indices of self where local max

        out_info : tuple (min_distance,smooth)
            min_distance used and bool indicating if smoothing was used


        """
        if num_phases_max is None:
            num_phases_max = self.num_phases_max

        if not isinstance(min_distance, Iterable):
            min_distance = [min_distance]

        if smooth is True:
            filtA = [True]
        elif smooth is False:
            filtA = [False]
        elif smooth.lower() == 'fail':
            filtA = [False, True]
        else:
            raise ValueError('bad parameter %s' % (smooth))

        for filt in filtA:
            if filt:
                xx = self.pad().smooth(**smooth_kwargs)
            else:
                xx = self

            for md in min_distance:
                x, n = xx._peak_local_max(
                    min_distance=md,
                    threshold_rel=0.00,
                    threshold_abs=0.2,
                    **kwargs)

                if n <= num_phases_max:
                    if info:
                        return x, (md, filt)
                    else:
                        return x

        #if got here than error
        raise RuntimeError('%i maxima found greater than %i at mu %s' %
                           (n, num_phases_max, repr(self.mu)))

    ##################################################
    #segmentation
    def get_labels_watershed(self,
                             indices,
                             structure='set',
                             size=3,
                             footprint=None,
                             smooth=False,
                             smooth_kwargs={},
                             **kwargs):
        """
        get labels from watershed segmentation

        Parameters
        ----------
        indices : tuple of ndarrays
         indices of location of points to segment to (e.g., maxima).
         len==ndim, seg_indices[i].shape == (nsegs)
         e.g., from output of self.peak_local_max

        structure : array or str or None
         structure passed to ndimage.label.
         if None, use default.
         if 'set', use np.one((3,)*data.ndim)
         else, use structure

        size : size parameter to _labels_watershed

        footprint : footprint parameter to _labels_watershed

        smooth : bool (Default False)
            if True, use smoothed data for label creation

        smooth_kwargs : dict
            arguments to self.smooth()

        **kwargs : dict
            arguments to _labels_watershed

        Returns
        -------
        labels
        """
        markers, n = _indices_to_markers(indices, self.shape, structure)
        if smooth:
            xx = self.pad().smooth(**smooth_kwargs)
        else:
            xx = self

        labels = _labels_watershed(
            -xx.data,
            markers,
            mask=(~self.mask),
            size=size,
            footprint=footprint,
            **kwargs)

        return labels

    def get_list_regmask(self, regmask, SegLenOne=True):
        """
        create list of lnPi objects with mask = self.mask + regmask[i]
        """
        if not SegLenOne and len(regmask) == 1:
            return None  #[self]
        else:
            return [self.add_mask(r) for r in regmask]

    def get_list_labels(self, labels, features=None, SegLenOne=True, **kwargs):
        """
        create list of lnpi's from labels
        """
        regmask = labels_to_masks(labels, features=features, **kwargs)
        return self.get_list_regmask(regmask, SegLenOne)

    def get_list_indices(self,
                         indices,
                         SegLenOne=True,
                         smooth=False,
                         smooth_kwargs={},
                         labels_kwargs={},
                         masks_kwargs={}):
        """
        create list of lnpi's from indices of features (argmax's)
        """
        if not SegLenOne and len(indices[0]) == 1:
            return None  #[self]
        else:
            labels = self.get_labels_watershed(
                indices,
                smooth=smooth,
                smooth_kwargs=smooth_kwargs,
                **labels_kwargs)
            return self.get_list_labels(
                labels, SegLenOne=SegLenOne, **masks_kwargs)

    @classmethod
    def from_table(cls,
                   path,
                   mu,
                   volume,
                   beta,
                   num_phases_max,
                   sep='\s+',
                   names=None,
                   csv_kws=None,
                   **kwargs):
        """
        Create lnPi object from text file table with columns [n_0,...,n_ndim, lnpi]

        Parameters
        ----------
        path : string like
            file object to be read
        mu : array-like
            chemical potential for each component
        volume : float
            total volume
        beta : float
            inverse temperature
        num_phases_max : int
        sep : string, optional
            separator for file read
        names : column names
        csv_kws : dict, optional
            optional arguments to `pandas.read_csv`
        kwargs  : extra arguments
            Passed to lnPi constructor
        """

        mu = np.atleast_1d(mu)
        ndim = len(mu)

        if names is None:
            names = ['n_{}'.format(i) for i in range(ndim)] + ['lnpi']

        if csv_kws is None:
            csv_kws = {}

        da = (pd.read_csv(path, sep=sep, names=names,
                          **csv_kws).set_index(names[:-1])['lnpi'].to_xarray())
        return cls(
            data=da.values,
            mask=da.isnull().values,
            mu=mu,
            volume=volume,
            beta=beta,
            num_phases_max=num_phases_max,
            **kwargs)

    # def split_phases(self,
    #                   argmax_kwargs=None,
    #                   phases_kwargs=None,
    #                   build_kwargs=None,
    #                   ftag_phases=None,
    #                   ftag_phases_kwargs=None):
    #     """
    #     return lnPi_phases object with placeholders for phases/argmax
    #     """
    #     raise ValueError


###################################################
### phases
###################################################


def concatify(key=None, prop=True, cache=False, **kws):
    def wrapper(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return xr.concat(func(self, *args, **kwargs), **kws)

        if cache:
            if prop:
                wrapped = cached(_key)(wrapped)
            else:
                wrapped = cached_func(_key)(wrapped)
        if prop:
            wrapped = property(wrapped)

        return wrapped

    return wrapper


concatify_phase = partial(concatify, dim='phase', coords='different')
concatify_rec = partial(concatify, dim='rec', coords='all')


def wrap(name, cache=True, prop=True, **kws):
    if prop:

        def _get_prop(self):
            return xr.concat([getattr(x, name) for x in self], **kws)
    else:

        def _get_prop(self, *args, **kwargs):
            return xr.concat([getattr(x, name)(*args, **kwargs) for x in self],
                             **kws)

    if cache:
        _get_prop = gcached(key=name, prop=prop)(_get_prop)
    elif prop:
        _get_prop = property(_get_prop)

    return _get_prop


def wrap_reindex(name, cache=True, prop=True, **kws):
    if prop:

        def _get_prop(self):
            return self._reindex(getattr(self, name))
    else:

        def _get_prop(self, *args, **kwargs):
            return self._reindex(getattr(self, name)(*args, **kwargs))

    if cache:
        _get_prop = gcached(key=name + "_phase", prop=prop)(_get_prop)
    elif prop:
        _get_prop = property(_get_prop)

    return _get_prop


def wrap_methods(methods, cache=True, prop=True, index=True, **kws):
    def wrap_methods_inner(cls):
        for name in methods:
            setattr(cls, name, wrap(name, cache, prop, **kws))
            if index:
                setattr(cls, name + '_phase',
                        wrap_reindex(name, cache, prop, **kws))
        return cls

    return wrap_methods_inner


@wrap_methods(
    ['nave', 'molfrac', 'nvar', 'density', 'lnpi', 'chempot', 'pi_norm'],
    prop=True,
    cache=True,
    dim='phase',
    coords='different')
@wrap_methods(['omega'],
              prop=False,
              cache=True,
              dim='phase',
              coords='different')
class Phases(object):
    """
    object containing lnpi base and phases
    """

    def __init__(
            self,
            base,
            phases='get',
            argmax='get',
            argmax_kwargs=None,
            phases_kwargs=None,
            build_kwargs=None,
            ftag_phases=None,
            ftag_phases_kwargs=None,
            phaseIDs=None,
    ):
        """
        object to store base and phases

        Parameters
        ----------
        base : lnPi object
            base object.  Not parsed to phases

        phases : list of lnPi objects, None, or str
            if list of lnPi objects, each corresponds to an independent phase.
            if None, implies single phase (phases=[base])
            if str and 'get', get phases on demand

        argmax : tuple of arrays (indicies) or str
            indices of local max locations.
            if str and 'get', get argmax on demand

        argmax_kwargs : dict
            if argmax=='get', use self.build_phases

        phases_kwargs : dict
            if phases=='get', use self.build_phases


        build_kwargs : dict
            arguments to self.buiild_phases

        ftag_phases : function (default tag_phases_binary)
            function which returns integer phase_id for each phase
            ...
            ftag_phases(self):
                returns [phase_id(i) for i in len(phases)]
        """

        self.base = base

        #self.phases = phases
        #self.argmax = argmax

        self._phaseIDs = phaseIDs

        if isinstance(ftag_phases, (bytes, str)):
            if ftag_phases == 'tag_phases_binary':
                ftag_phases = tag_phases_binary
            elif ftag_phases == 'tag_phases_single':
                ftag_phases = tag_phases_single
            else:
                raise ValueError(
                    'if specify with string, but be in ["tag_phases_binary", "tag_phases_single"]'
                )

        self._ftag_phases = ftag_phases

        if argmax_kwargs is None:
            argmax_kwargs = {}
        if phases_kwargs is None:
            phases_kwargs = {}
        if build_kwargs is None:
            build_kwargs = {}
        if ftag_phases_kwargs is None:
            ftag_phases_kwargs = {}

        self._argmax_kwargs = argmax_kwargs
        self._phases_kwargs = phases_kwargs
        self._build_kwargs = build_kwargs
        self._ftag_phases_kwargs = ftag_phases_kwargs

        if phases == 'get':
            self._phases = phases
            # if have argmax, use it
            if argmax == 'get':
                self._argmax = argmax
                self.build_phases(inplace=True, **self._build_kwargs)
            else:
                self.argmax = argmax
                self.phases = self.base.get_list_indices(
                    self.argmax, **self._phase_kwargs)
        else:
            self.phases = phases
            if argmax == 'get':
                # set argmax from phases
                self.argmax = self.argmax_from_phases()

    ##################################################
    #copy
    def copy(self, **kwargs):
        """
        create shallow copy of self

        **kwargs : named arguments to lnPi_collection.__init__
        if argument is given, it will overide that in self
        """

        d = {}
        for k in [
                'base', 'phases', 'argmax', 'argmax_kwargs', 'phases_kwargs',
                'build_kwargs', 'ftag_phases', 'ftag_phases_kwargs'
        ]:
            if k in kwargs:
                d[k] = kwargs[k]
            else:
                _k = '_' + k
                d[k] = getattr(self, _k)

        return self.__class__(**d)

    ##################################################
    #reweight
    def reweight(self,
                 mu,
                 zeromax=True,
                 pad=False,
                 phases='get',
                 argmax='get',
                 **kwargs):
        """
        create a new lnpi_phases reweighted to new mu
        """

        return self.copy(
            base=self.base.reweight(mu, zeromax=zeromax, pad=pad, **kwargs),
            phases=phases,
            argmax=argmax)

    ##################################################
    #properties
    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        if type(val) is not lnPi:
            raise ValueError('base must be type lnPi %s' % (type(val)))
        self._base = val

    @property
    def argmax(self):
        return self._argmax

    @argmax.setter
    def argmax(self, val):
        assert self.base.ndim == len(val)
        assert len(val[0]) <= self.base.num_phases_max
        self._argmax = val

    def argmax_from_phases(self):
        """
        set argmax from max in each phase
        """
        L = [np.array(np.where(p.filled() == p.max())).T for p in self.phases]
        argmax = tuple(np.concatenate(L).T)
        return argmax

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, phases):
        if isinstance(phases, (tuple, list)):
            #assume that have list/tuple of lnPi phases
            for i, p in enumerate(phases):
                if type(p) is not lnPi:
                    raise ValueError('element %i of phases must be lnPi' % (i))

                if np.any(self.base.mu != p.mu):
                    raise ValueError('bad mu between base and comp %i: %s, %s'% \
                                     (i,self.base.mu,p.mu))

        elif isinstance(phases, np.ndarray):
            #assume labels:
            phases = self.base.get_list_labels(phases)
        elif phases is None:
            phases = [self.base]
        else:
            raise ValueError(
                'phases must be a list of lnPi, label array, None, or "get"')

        self._phases = phases

    def __len__(self):
        return len(self.argmax[0])

    def __getitem__(self, i):
        return self.phases[i]

    @property
    def nphase(self):
        return len(self)

    def _tag_phases(self):
        if self._ftag_phases is None:
            if self._phaseIDs is not None:
                return np.array(self._phaseIDs)
            else:
                return np.arange(len(self))
        else:
            return self._ftag_phases(self, **self._ftag_phases_kwargs)

    @property
    def phaseIDs(self):
        return self._tag_phases()

    def phaseIDs_to_indicies(self, IDs):
        """
        convert IDs to indicies in self.phases
        """
        l = list(self.phaseIDs)
        return np.array([l.index(i) for i in IDs])

    @property
    def has_phaseIDs(self):
        """
        return array of bools which are True if index=phaseID is present
        """
        b = np.zeros(self.base.num_phases_max, dtype=bool)
        b[self.phaseIDs] = True
        return b

    @property
    def masks(self):
        return np.array([p.mask for p in self.phases])

    @property
    def labels(self):
        return masks_to_labels(
            self.masks, feature_value=False, values=self.phaseIDs)

    def _reindex(self, x):
        return x.assign_coords(phase=self.phaseIDs).reindex(
            phase=range(self.base.num_phases_max))

    @property
    def mu(self):
        return self.base.mu

    @property
    def beta(self):
        return self.base.beta

    @property
    def volume(self):
        return self.base.volume

    @property
    def ncoords(self):
        return self.base.coords

    ##################################################
    #query
    def _get_boundaries(self, IDs, mode='thick', connectivity=None, **kwargs):
        """
        get the boundary between phase pair

        Parameters
        ----------
        IDs : iterable of phases indices to get boundaries about

        mode : string (Default 'thick')
         mode passed to find_boundaries

        connectivity : int (Default None)
         if None, use self.ndim

        **kwargs : extra arguments to find_boundaries

        Returns
        -------
        output : array of shape self.base.shape of bools
          output==True at boundary locations
        """

        if connectivity is None:
            connectivity = self.base.ndim
        b = []
        for i in IDs:
            p = self.phases[i]
            msk = np.atleast_2d((~p.mask).astype(int))
            b.append(
                find_boundaries(
                    msk, mode=mode, connectivity=connectivity, **kwargs))
        return b

    def _get_boundaries_overlap(self,
                                IDs,
                                mode='thick',
                                connectivity=None,
                                **kwargs):
        """
        get overlap between phases in IDs
        """

        boundaries = self._get_boundaries(
            IDs, mode=mode, connectivity=connectivity, **kwargs)

        #loop over all combinations
        ret = {}
        for i, j in itertools.combinations(range(len(IDs)), 2):

            b = np.prod([boundaries[i],boundaries[j]],axis=0)\
                  .astype(bool).reshape(self.base.shape) * \
                  (~self.base.mask)

            if b.sum() == 0:
                b = None

            ret[i, j] = b

        return ret

    @property
    def betaEmin(self):
        """
        betaE_min = -max{lnPi}
        """
        return -np.array([p.max() for p in self.phases])

    def betaEtransition(self, IDs, **kwargs):
        """
        minimum value of energy at boundary between phases
        beta E_boundary = - max{lnPi}_{along boundary}

        if no boundary found between phases (i.e., they are not connected),
        then return vmax
        """

        boundaries = self._get_boundaries_overlap(IDs, **kwargs)

        ret = {}
        for k in boundaries:
            b = boundaries[k]

            if b is None:
                ret[k] = np.inf
            else:
                ret[k] = -(self.base[b].max())

        return ret

    def betaEtransition_matrix(self, **kwargs):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        ET = self.betaEtransition(range(self.nphase), **kwargs)

        out = np.full((self.nphase, ) * 2, dtype=float, fill_value=np.inf)
        for i, j in ET:
            out[i, j] = out[j, i] = ET[i, j]

        return out

    def DeltabetaE_matrix(self, vmax=1e20, **kwargs):
        """
        out[i,j]=DeltaE between phase[i] and phase[j]

        if no transition between (i,j) , out[i,j] = vmax
        """
        out = self.betaEtransition_matrix(**kwargs) - \
             self.betaEmin[:,None]

        #where nan and not on diagonal, set to vmax
        out[np.isnan(out)] = vmax
        np.fill_diagonal(out, np.nan)

        return out

    def DeltabetaE_matrix_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        """
        out[i,j] =Delta E between phaseID==i and phaseID==j

        if i does not exist, then out[i,j] = vmin
        if not transition to j, then out[i,j] = vmax
        """

        dE = self.DeltabetaE_matrix(vmax, **kwargs)
        out = np.empty((self.base.num_phases_max, ) * 2, dtype=float) * np.nan
        phaseIDs = self.phaseIDs

        for i, ID in enumerate(phaseIDs):
            out[ID, phaseIDs] = dE[i, :]

        #where nan fill in with vmax
        out[np.isnan(out)] = vmax
        #where phase does not exist, fill with vmin
        has_phaseIDs = self.has_phaseIDs
        out[~has_phaseIDs, :] = vmin
        np.fill_diagonal(out, np.nan)

        return out

    def DeltabetaE_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        """
        minimum transition energy from phaseID to any other phase
        """
        return np.nanmin(
            self.DeltabetaE_matrix_phaseIDs(vmin, vmax, **kwargs), axis=-1)

    def _betaEtransition_line(self, pair=(0, 1), connectivity=None):
        """
        find transition energy from line connecting location of maxima
        """

        args = tuple(np.array(self.argmax).T[pair, :].flatten())

        img = np.zeros(self.base.shape, dtype=int)
        img[draw.line(*args)] = 1

        if connectivity is None:
            msk = img.astype(bool)
        else:
            msk = find_boundaries(img, connectivity=connectivity)

        return -(self.base[msk].min())

    def _betaEtransition_path(self, pair=(0, 1), **kwargs):
        """
        construct a path connecting maxima
        """

        idx = np.array(self.argmax).T

        d = -self.base.data.copy()
        d[self.base.mask] = d.max()

        i, w = route_through_array(d, idx[pair[0]], idx[pair[1]], **kwargs)

        i = tuple(np.array(i).T)

        return -(self.base[i].min())

    def sort_by_phaseIDs(self, inplace=False):
        """
        sort self so that self.phaseIDs are increasing
        """

        idx = np.argsort(self.phaseIDs)
        argmax = tuple(x[idx] for x in self.argmax)
        argmax = tuple(np.array(self.argmax).T[idx, :].T)
        L = [self._phases[i] for i in idx]

        if inplace:
            self.argmax = argmax
            self.phases = L
        else:
            return self.copy(phases=L)

    ##################################################
    #repr
    def _repr_html_(self):
        if self._phases == 'get':
            x = self._phases
        else:
            x = self.nphase
        return 'Phases: nphase=%s, mu=%s' % (x, self.base.mu)

    @staticmethod
    def _get_DE(Etrans, Emin, vmax=1e20):
        DE = Etrans - Emin[:, None]
        DE[np.isnan(DE)] = vmax
        np.fill_diagonal(DE, np.nan)
        return DE

    def merge_phases(self,
                     efac=1.0,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     **kwargs):
        """
        merge phases such that DeltabetaE[i,j]>efac-tol for all phases

        Parameters
        ----------
        efac : float (Default 1.0)
            cutoff

        vmax : float (1e20)
            vmax parameter in DeltabetaE calculation

        inplace : bool (Default False)
            if True, do inplace modificaiton, else return

        force : bool (Default False)
            if True, iteratively remove minimum energy difference
            (even if DeltabetaE>efac) until have
            exactly self.base.num_phases_max


        **kwargs : arguments to DeltabetE_matrix


        Return
        ------
        out : Phases object (if inplace is False)
        """

        if self._phases is None:
            if inplace:
                return
            else:
                return self.copy()

        #first get Etrans,Emin
        Etrans = self.betaEtransition_matrix()
        Emin = self.betaEmin

        #will iteratively reduce dE=Etrans - Emin
        #keep placeholder of absolute indices still in dE (number)
        #as well as list of phases merged together (L)
        #where L==None, phase has been removed.  removed phases
        #are appended to L[i] where i is the phase merged into.
        number = np.arange(Emin.shape[0])
        L = [[i] for i in range(len(Emin))]

        while True:
            nphase = len(Emin)
            if nphase == 1:
                break
            DE = self._get_DE(Etrans, Emin, vmax=vmax)

            min_val = np.nanmin(DE)
            if min_val > efac:
                if not force:
                    break
                elif nphase <= self.base.num_phases_max:
                    break

            idx = np.where(DE == min_val)
            idx_kill, idx_keep = idx[0][0], idx[1][0]

            #add in new row taking min of each transition energy
            new_row = np.nanmin(Etrans[[idx_kill, idx_keep], :], axis=0)
            new_row[idx_keep] = np.nan

            #add in new row/col
            Etrans[idx_keep, :] = new_row
            Etrans[:, idx_keep] = new_row

            #delete idx_kill
            Etrans = np.delete(np.delete(Etrans, idx_kill, 0), idx_kill, 1)
            Emin = np.delete(Emin, idx_kill)

            #update L
            L[number[idx_keep]] += L[number[idx_kill]]
            L[number[idx_kill]] = None
            #update number
            number = np.delete(number, idx_kill)

        msk = np.array([x is not None for x in L])

        argmax_new = tuple(x[msk] for x in self._argmax)
        phases_new = []

        for x in L:
            if x is None: continue
            mask = np.all([self._phases[i].mask for i in x], axis=0)
            phases_new.append(self.base.new_mask(mask))

        if inplace:
            self.argmax = argmax_new
            self.phases = phases_new
        else:
            return self.copy(phases=phases_new, argmax=argmax_new)

    def merge_phaseIDs(self, inplace=False, **kwargs):
        """
        if phaseIDs are equal, merge them to a single phase
        """

        if inplace:
            new = self
        else:
            new = self.copy()

        phaseIDs = self._tag_phases()
        if len(phaseIDs) > 0:
            #find distance between phaseIDs
            dist = pdist(phaseIDs.reshape(-1, 1)).astype(int)
            if np.any(dist == 0):
                #have some phases with same phaseIDs.

                #create list of where
                L = []
                for i in range(self.base.num_phases_max):
                    w = np.where(phaseIDs == i)[0]
                    if len(w) > 0:
                        L.append(w)

                #merge phases
                phases_new = []
                for x in L:
                    if len(x) == 1:
                        #only one phase has this phase
                        phases_new.append(self.phases[i])
                    else:
                        #multiple phases have this phaseID
                        mask = np.all([self.phases[i].mask for i in x], axis=0)
                        phases_new.append(self.base.new_mask(mask))

                new.phases = phases_new
                new.argmax = new.argmax_from_phases()
                if hasattr(self, '_phaseIDs'):
                    del self._phaseIDs

        if not inplace:
            return new

    def build_phases(self,
                     merge='full',
                     nmax_start=10,
                     efac=0.1,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     merge_phaseIDs=True,
                     **kwargs):
        """
        iteratively build phases by finding argmax merging phases

        Parameters
        ----------
        merge : str or None (Default 'partial')
            if None, don't perform merge.
            if 'full', perform normal merge with passed `force` and `efac`
            if 'partial' -> only merge if necessary and with force=True, efac=0.0
            (i.e., will leave phases with dE<`efac`)


        nmax_start : int (Default 10)
            max number of phases argmax_local to start with


        efac : float (Default 0.2)
            merge all phases where DeltabetaE_matrix()[i,j] < efac-tol

        vmax : float (Default 1e20)
            value of DeltabetaE if phase transition between i and j does not exist


        inplace : bool (Default False)
            if True, do inplace modification

        force : bool (Default True)
            if True, keep removing the minimum transition phases regardless of dE until
            nphase==num_phases_max

        merge_phaseIDs : bool (Default True)
            if True, merge phases with same phaseIDs

        **kwargs : extra arguments to self.merge_phases

        Returns
        -------
        out : Phases (optional, if inplace is False)
        """

        #print 'building phase'

        if inplace:
            t = self
        else:
            t = self.copy()

        if t._argmax == 'get':
            #use _argmax to avoid num_phases_max check
            t._argmax = t._base.argmax_local(
                num_phases_max=nmax_start, **t._argmax_kwargs)

        if t._phases == 'get':
            #use _phases to avoid checks
            t._phases = self._base.get_list_indices(t._argmax,
                                                    **t._phases_kwargs)

        if t.nphase == 1:
            #nothing to do here
            pass

        elif merge is None:
            pass

        elif merge == 'full':
            t.merge_phases(
                efac=efac, vmax=vmax, inplace=True, force=force, **kwargs)

        elif merge == 'partial':
            t.merge_phases(
                efac=0.0, vmax=vmax, inplace=True, force=True, **kwargs)

        #do a sanity check
        t.argmax = t._argmax
        t.phases = t._phases

        if merge_phaseIDs:
            t.merge_phaseIDs(inplace=True)

        t.argmax = t.argmax_from_phases()

        if not inplace:
            return t

    def to_dataarray(self, dtype=np.uint8, **kwargs):
        """
        create dataarray object from labels
        """

        data = self.labels
        if dtype is not None:
            data = data.astype(dtype)
        return xr.DataArray(
            data,
            dims=self.base.dims_n,
            name='labels',
            coords=self.base.coords_state)

    @classmethod
    def from_dataarray(cls, base, da, **kwargs):

        labels = da.values
        ndim = labels.ndim

        mu = [da.coords[k] * 1.0 for k in base.dims_mu]

        beta = da.coords['beta']
        volume = da.coords['volume']

        assert beta == base.beta
        assert volume == base.volume

        return cls.from_labels(base=base, labels=labels, mu=mu, **kwargs)

    @classmethod
    def from_dataarray_groupby(cls, base, da, dim='rec', **kwargs):

        lnpis = []
        for i, g in da.groupby(dim):
            lnpis.append(cls.from_dataarray(base=base, da=g, **kwargs))
        return lnpis

    @classmethod
    def from_labels(cls,
                    base,
                    labels,
                    mu=None,
                    argmax='get',
                    SegLenOne=True,
                    mask_kwargs=None,
                    **kwargs):
        """
        create Phases from labels
        """
        # TODO
        if mu is not None:
            base = base.reweight(mu)
        if mask_kwargs is None:
            mask_kwargs = {}

        features = np.unique(labels)
        features = features[features > 0]

        phaseIDs = features - 1

        phases = base.get_list_labels(
            labels, features=features, SegLenOne=SegLenOne, **mask_kwargs)
        new = cls(
            base=base,
            phases=phases,
            argmax=argmax,
            phaseIDs=phaseIDs,
            **kwargs)
        if argmax == 'get':
            new.argmax = new.argmax_from_phases()
        return new


################################################################################
#collection
################################################################################
#@wrap_methods(['chempot'], cache=True, prop=True, index=False, coords='all', dim='rec')
@wrap_methods(
    [x + '_phase' for x in ('chempot', 'nave', 'nvar', 'molfrac', 'density')],
    cache=True,
    prop=True,
    index=False,
    coords='all',
    dim='rec')
@wrap_methods(['omega_phase'],
              cache=True,
              prop=False,
              index=False,
              coords='all',
              dim='rec')
class Collection(object):
    """
    class containing several lnPis
    """

    def __init__(self, lnpis):
        """
        Parameters
        ----------
        lnpis : iterable of lnpi_phases objects
        """
        self.lnpis = lnpis

    ##################################################
    #copy
    def copy(self, lnpis=None):
        """
        create shallow copy of self

        **kwargs : named arguments to Collection.__init__
        if argument is given, it will overide that in self
        """
        if lnpis is None:
            lnpis = self.lnpis[:]

        return Collection(lnpis=lnpis)

    ##################################################
    #setters
    def _parse_lnpi(self, x):
        if type(x) is not Phases:
            raise ValueError('bad value while parsing element %s' % (type(x)))
        else:
            return x

    def _parse_lnpis(self, lnpis):
        """from a list of lnpis, return list of lnpi_phases"""
        if not isinstance(lnpis, (list, tuple)):
            raise ValueError('lnpis must be list or tuple')

        return [self._parse_lnpi(x) for x in lnpis]

    ##################################################
    #properties
    @property
    def lnpis(self):
        return self._lnpis

    @lnpis.setter
    @cached_clear()
    def lnpis(self, val):
        self._lnpis = self._parse_lnpis(val)

    ##################################################
    #list props
    def __len__(self):
        return len(self.lnpis)

    @property
    def shape(self):
        return (len(self), )

    @cached_clear()
    def append(self, val, unique=True, decimals=5):
        """append a value to self.lnpis"""
        if unique:
            if len(self._unique_mus(val.mu, decimals)) > 0:
                self._lnpis.append(self._parse_lnpi(val))
        else:
            self._lnpis.append(self._parse_lnpi(val))

    @cached_clear()
    def extend(self, x, unique=True, decimals=5):
        """extend lnpis"""
        if isinstance(x, Collection):
            x = x._lnpis

        if isinstance(x, list):
            if unique:
                x = self._unique_list(x, decimals)
            self._lnpis.extend(self._parse_lnpis(x))
        else:
            raise ValueError('only lists or Collections can be added')

    def extend_by_mu_iter(self, ref, mus, unique=True, decimals=5, **kwargs):
        """extend by mus"""
        if unique:
            mus = self._unique_mus(mus, decimals=decimals)
        new = Collection.from_mu_iter(ref, mus, **kwargs)
        self.extend(new, unique=False)

    def extend_by_mu(self, ref, mu, x, unique=True, decimals=5, **kwargs):
        """extend self my mu values"""
        mus = get_mu_iter(mu, x)
        self.extend_by_mu_iter(ref, mus, unique, decimals, **kwargs)

    def __add__(self, x):
        if isinstance(x, list):
            L = self._lnpis + self._parse_lnpis(x)
        elif isinstance(x, Collection):
            L = self._lnpis + x._lnpis
        else:
            raise ValueError('only lists or Collections can be added')

        return self.copy(lnpis=L)

    def __iadd__(self, x):
        if isinstance(x, list):
            L = self._parse_lnpis(x)
        elif isinstance(x, Collection):
            L = x._lnpis
        else:
            raise ValueError('only list or Collections can be added')

        self._lnpis += L
        return self

    def sort_by_mu(self, comp=0, inplace=False):
        """
        sort self.lnpis by mu[:,comp]
        """
        order = np.argsort(self.mus[:, comp])
        L = [self._lnpis[i] for i in order]
        if inplace:
            self._lnpis = L
            self._cache = {}
        else:
            return self.copy(lnpis=L)

    def _unique_list(self, L, decimals=5):
        """
        limit list such that output[i].mu not in self.mus
        """

        tol = 0.5 * 10**(-decimals)
        mus = np.array([x.mu for x in L])
        new = np.atleast_2d(mus)

        msk = np.all(cdist(self.mus, new) > tol, axis=0)

        return [x for x, m in zip(L, msk) if m]

    def _unique_mus(self, mus, decimals=5):
        """
        return only those mus not already in self

        Parameters
        ----------
        mus : arrray of new mus
            shape is (ncomp,) or (m,ncomp). make 2d if not already

        decimals : int (Default 5)
            consider mu replicated if dist between any mu already in
            self and mus[i] <0.5*10**(-decimals)

        Returns
        -------
        output : bool or array of bools
        """

        tol = 0.5 * 10**(-decimals)
        mus = np.asarray(mus)
        new = np.atleast_2d(mus)
        msk = np.all(cdist(self.mus, new) > tol, axis=0)
        return new[msk, :]

    @cached_clear()
    def drop_duplicates(self, decimals=5):
        """
        drop doubles of given mu
        """
        tol = 0.5 * 10**(-decimals)

        mus = self.mus
        msk = squareform(pdist(mus)) < tol
        np.fill_diagonal(msk, False)

        a, b = np.where(msk)
        b = b[b > a]

        keep = np.ones(mus.shape[0], dtype=bool)
        keep[b] = False

        self._lnpis = [x for x, m in zip(self._lnpis, keep) if m]

    def __getitem__(self, i):
        if isinstance(i, (np.int, np.integer)):
            return self.lnpis[i]

        elif type(i) is slice:
            L = self.lnpis[i]

        elif isinstance(i, (list, np.ndarray)):
            idx = np.array(i)

            if np.issubdtype(idx.dtype, np.integer):
                L = [self.lnpis[j] for j in idx]

            elif np.issubdtype(idx.dtype, np.bool):
                assert idx.shape == self.shape
                L = [xx for xx, mm in zip(self.lnpis, idx) if mm]

            else:
                raise KeyError('bad key')

        return self.copy(lnpis=L)

    def merge_phases(self,
                     efac=1.0,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     **kwargs):
        L = [
            x.merge_phases(
                efac=efac, vmax=vmax, inplace=inplace, force=force, **kwargs)
            for x in self._lnpis
        ]
        if not inplace:
            return self.copy(lnpis=L)
        else:
            self._cache = {}

    ##################################################
    #calculations/props
    @property
    def mus(self):
        return np.array([x.mu for x in self.lnpis])

    @property
    def nphases(self):
        return np.array([x.nphase for x in self.lnpis])

    @property
    def has_phaseIDs(self):
        return np.array([x.has_phaseIDs for x in self.lnpis])

    def DeltabetaE_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        return np.array(
            [x.DeltabetaE_phaseIDs(vmin, vmax, **kwargs) for x in self])

    ##################################################
    #spinodal
    def get_spinodal_phaseID(self,
                             ID,
                             efac=1.0,
                             dmu=0.5,
                             vmin=0.0,
                             vmax=1e20,
                             ntry=20,
                             step=None,
                             nmax=20,
                             reweight_kwargs={},
                             DeltabetaE_kwargs={},
                             close_kwargs={},
                             solve_kwargs={},
                             full_output=False):
        """
        locate spinodal for phaseID ID
        """
        s, r = get_spinodal(
            self,
            ID,
            efac=efac,
            dmu=dmu,
            vmin=vmin,
            vmax=vmax,
            ntry=ntry,
            step=step,
            nmax=nmax,
            reweight_kwargs=reweight_kwargs,
            DeltabetaE_kwargs=DeltabetaE_kwargs,
            close_kwargs=close_kwargs,
            solve_kwargs=solve_kwargs,
            full_output=True)

        if full_output:
            return s, r
        else:
            return s

    def get_spinodals(
            self,
            efac=1.0,
            dmu=0.5,
            vmin=0.0,
            vmax=1e20,
            ntry=20,
            step=None,
            nmax=20,
            reweight_kwargs={},
            DeltabetE_kwargs={},
            close_kwargs={},
            solve_kwargs={},
            inplace=True,
            append=True,
            force=False,
    ):

        if inplace and hasattr(self, '_spinodals') and not force:
            raise ValueError('already set spinodals')

        L = []
        info = []
        for ID in range(self[0].base.num_phases_max):
            s, r = self.get_spinodal_phaseID(
                ID,
                efac=efac,
                dmu=dmu,
                vmin=vmin,
                vmax=vmax,
                ntry=ntry,
                step=step,
                nmax=nmax,
                reweight_kwargs=reweight_kwargs,
                DeltabetaE_kwargs=DeltabetE_kwargs,
                close_kwargs=close_kwargs,
                solve_kwargs=solve_kwargs,
                full_output=True)

            L.append(s)
            info.append(r)

        if append:
            for x in L:
                if x is not None:
                    self.append(x)

        if inplace:
            self._spinodals = L
            self._spinodals_info = info
        else:
            return L, info

    @property
    def spinodals(self):
        if not hasattr(self, '_spinodals'):
            raise AttributeError('spinodal not set')

        return self._spinodals

    ##################################################
    #binodal
    def get_binodal_pair(self,
                         IDs,
                         spinodals=None,
                         reweight_kwargs={},
                         full_output=False,
                         **kwargs):

        if spinodals is None:
            spinodals = self.spinodals
        spin = [self.spinodals[i] for i in IDs]

        if None in spin:
            #raise ValueError('one of spinodals is Zero')
            b, r = None, None
        else:
            b, r = get_binodal_point(
                self[0],
                IDs,
                spin[0].mu,
                spin[1].mu,
                reweight_kwargs=reweight_kwargs,
                full_output=True,
                **kwargs)

        if full_output:
            return b, r
        else:
            return b

    def get_binodals(self,
                     spinodals=None,
                     reweight_kwargs={},
                     inplace=True,
                     append=True,
                     force=False,
                     **kwargs):

        if inplace and hasattr(self, '_binodals') and not force:
            raise ValueError('already set spinodals')

        if spinodals is None:
            spinodals = self.spinodals

        L = []
        info = []
        for IDs in itertools.combinations(
                range(self[0].base.num_phases_max), 2):

            b, r = self.get_binodal_pair(
                IDs,
                spinodals,
                reweight_kwargs=reweight_kwargs,
                full_output=True,
                **kwargs)

            L.append(b)
            info.append(r)

        if append:
            for x in L:
                if x is not None:
                    self.append(x)

        if inplace:
            self._binodals = L
            self._binodals_info = info

        else:
            return L, info

    @property
    def binodals(self):
        if not hasattr(self, '_binodals'):
            raise AttributeError('binodals not set')

        return self._binodals

    def get_binodal_interp(self, mu_axis, IDs=(0, 1)):
        """
        get position of Omega[i]==Omega[j] by interpolation
        """

        idx = np.asarray(IDs)

        assert (len(idx) == 2)

        x = self.Omegas_phaseIDs()[:, idx]
        msk = np.prod(~np.isnan(x), axis=1).astype(bool)
        assert (msk.sum() > 0)

        diff = x[msk, 0] - x[msk, 1]

        mu = self.mus[msk, mu_axis]
        i = np.argsort(diff)

        return np.interp(0.0, diff[i], mu[i])

    ##################################################
    def _repr_html_(self):
        return 'Collection: %s' % len(self)

    ##################################################
    #builders
    ##################################################
    @classmethod
    def from_mu_iter(cls, ref, mus, **kwargs):
        """
        build Collection from mus

        Parameters
        ----------
        ref : lnpi_phases object
            lnpi_phases to reweight to get list of lnpi's

        mus : iterable
            chem. pots. to get lnpi

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : Collection object
        """

        assert isinstance(ref, Phases)

        kwargs = dict(dict(zeromax=True), **kwargs)

        L = [ref.reweight(mu, **kwargs) for mu in mus]

        return cls(L)

    @classmethod
    def from_mu(cls, ref, mu, x, **kwargs):
        """
        build Collection from mu builder

        Parameters
        --------- 
        ref : lnpi object
            lnpi to reweight to get list of lnpi's

        mu : list
            list with one element equal to None.
            This is the component which will be varied
            For example, mu=[mu0,None,mu2] implies use values
            of mu0,mu2 for components 0 and 2, and vary component 1

        x : array
            values to insert for variable component

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : Collection object
        """

        mus = get_mu_iter(mu, x)
        return cls.from_mu_iter(ref, mus, **kwargs)

    def to_dataarray(self,
                     dim='rec',
                     coords='all',
                     dtype=np.uint8,
                     phase_kws=None,
                     **kwargs):

        if phase_kws is None:
            phase_kws = {}
        # dataarray of labels
        da = xr.concat(
            [x.to_dataarray(dtype=dtype, **phase_kws) for x in self],
            dim=dim,
            coords=coords,
            **kwargs)

        # add in spinodal/binodal
        for k in ['spinodals', 'binodals']:
            _k = '_' + k
            label = np.zeros(len(self), dtype=dtype)
            if hasattr(self, _k):
                for i, target in enumerate(getattr(self, _k)):
                    if target is None:
                        break
                    for rec, x in enumerate(self):
                        if x is target:
                            label[rec] = i + 1
            # else no mark
            da.coords[k] = (dim, label)
        return da

    @classmethod
    def from_dataarray(cls, base, da, dim='rec', child=Phases, child_kws=None):

        if child_kws is None:
            child_kws = {}
        lnpis = child.from_dataarray_groupby(
            base=base, da=da, dim=dim, **child_kws)
        new = cls(lnpis)

        d = {}
        for k in ['spinodals', 'binodals']:
            _k = '_' + k
            label = da.coords[k]
            features = np.unique(label[label > 0])
            for feature in features:
                idx = np.where(label == feature)[0][0]
                if _k not in d:
                    d[_k] = [lnpis[idx]]
                else:
                    d[_k].append(lnpis[idx])
        for _k, v in d.items():
            if len(v) > 0:
                setattr(new, _k, v)
        return new

        # if 'rec_spinodals' in da.attrs:
        #     new._spinodals = []
        #     for i in np.atleast_1d(da.attrs['rec_spinodals']):
        #         new._spinodals.append(lnpis[i])

        # if 'rec_binodals' in da.attrs:
        #     new._binodals = []
        #     for i in np.atleast_1d(da.attrs['rec_binodals']):
        #         new._binodals.append(lnpis[i])
        # return new
