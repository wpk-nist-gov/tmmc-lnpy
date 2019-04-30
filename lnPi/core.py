import numpy as np
import xarray as xr


from functools import wraps, partial

from scipy.ndimage import filters

from lnPi.cached_decorators import cached_clear, cached, cached_func
from lnPi._utils import _interp_matrix, get_mu_iter

# Decorators
def gcached(key=None, prop=True):
    def wrapper(func):
        if prop:
            wrapped = property(cached(key)(func))
        else:
            wrapped = cached_func(key)(func)
        return wrapped
    return wrapper


def xrify(key=None, prop=True, cache=False, **kws):
    def wrapper(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return xr.DataArray(func(self, *args, **kwargs), coords=self._xr_coords, name=_key, **kws)

        if cache:
            if prop:
                wrapped = cached(_key)(wrapped)
            else:
                wrapped = cached_func(_key)(wrapped)
        if prop:
            wrapped = property(wrapped)

        return wrapped
    return wrapper
xrify_comp = partial(xrify, dims='component')


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

    Nave : average number of particles of each component    divider = make_axes_locatable(ax)
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

    adjust : ZeroMax and/or Pad


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

        fv = kwargs.get('fill_value', None) or getattr(data, 'fill_value', None)
        if fv is not None:
            obj.set_fill_value(fv)

        obj._optinfo.update(mu=mu, num_phases_max=num_phases_max, volume=volume, beta=beta)
        obj.adjust(zeromax=zeromax, pad=pad, inplace=True)

        return obj

    ##################################################
    #caching
    def __array_finalize__(self, obj):
        super(lnPi, self).__array_finalize__(obj)
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


    @gcached()
    def _xr_coords(self):
        coords = {'mu_{}'.format(i) : v for i,v in enumerate(self.mu)}
        coords['beta'] = self.beta
        return coords

    @gcached()
    def dims_N(self):
        return ['N_{}'.format(i) for i in range(self.ndim)]

    @gcached()
    def ncoords(self):
        """particle number for each particle dimension"""
        return xr.DataArray(np.indices(self.shape), dims=['component'] + self.dims_N, coords=self._xr_coords)

    @xrify_comp(cache=True)
    def chempot(self):
        return self.mu

    #calculated properties
    @gcached()
    def pi(self):
        """
        basic pi = exp(lnpi)
        """
        shift = self.max()
        pi = np.exp(self.filled(np.nan) - shift)
        return xr.DataArray(pi, dims=self.dims_N, coords=self._xr_coords)

    @gcached()
    def pi_sum(self):
        return self.pi.sum()

    @gcached()
    def pi_norm(self):
        p = self.pi
        return p / self.pi_sum

    @gcached()
    def nave(self):
        """average number of particles of each component"""
        return (self.pi_norm * self.ncoords).sum(self.dims_N)

    @property
    def density(self):
        """density of each component"""
        return self.nave / self.volume

    @gcached()
    def nvar(self):
        return (self.pi_norm * (self.ncoords - self.nave)**2).sum(self.dims_N)

    @property
    def molfrac(self):
        n = self.nave
        return n / n.sum()

    @xrify(cache=True, prop=False)
    def omega(self, *args):
        """
        get omega = zval - ln(sum(pi))

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """

        if len(args) == 0:
            zval = None
        else:
            zval = args[0]

        if zval is None:
            zval = self.data.ravel()[0] - self.max()

        omega = (zval - np.log(self.pi_sum)) / self.beta
        return omega

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

        ZeroMax : bool (Default False)

        Pad : bool (Default False)

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
        return lnPi(self.data, mask=mask, **dict(self._optinfo, **kwargs))

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


    def to_dataarray(self, **kwargs):
        """
        create a xarray.DataArray from self.

        Parameters
        ----------
        kwargs : extra arguments to xarray.DataArray

        if rec is not None, insert a record dimension in front
        """


        coords = dict(self._xr_coords, mask=(self.dims_N, self.mask))

        coords['volume'] = self.volume or np.nan
        coords['num_phases_max'] = self.num_phases_max or np.nan

        return xr.DataArray(self.data, dims=self.dims_N, coords=coords, **kwargs)

    @classmethod
    def from_dataarray(cls, da, **kwargs):
        """
        create a lnPi object from xarray.DataArray
        """

        data = da.values
        mask = da.mask.values

        ndim = data.ndim

        kws = {}
        kws['mu'] = []
        for k, v  in da.coords.items():
            if k == 'mask':
                continue
            v = v.values * 1
            if 'mu' in k:
                kws['mu'].append(v)
            else:
                kws[k] = v
        kwargs = dict(kws, **kwargs)

        return cls(data, mask=mask, **kwargs)





