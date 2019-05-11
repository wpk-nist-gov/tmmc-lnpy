from __future__ import print_function, absolute_import, division

import numpy as np
import xarray as xr
import pandas as pd

from scipy.ndimage import filters

from .cached_decorators import gcached


# NOTE : This is a rework of core.
# [ ] : split xarray functionality into wrapper(s)
# [ ] : split splitting into separate classes



class MaskedlnPi(np.ma.MaskedArray):
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

        obj._optinfo.update(
            mu=mu, volume=volume, beta=beta)
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
    def volume(self):
        return self._optinfo.get('volume', None)

    @property
    def beta(self):
        return self._optinfo.get('beta', None)

    def __setitem__(self, index, value):
        self._clear_cache()
        super().__setitem__(index, value)

    # accessors
    @gcached()
    def xtm(self):
#         if not hasattr(self, 'xtm'):
#             self._xtm = xrlnPi(self)
#         return self._xtm
        return xrlnPi(self)

    @property
    def xlnpi(self):
        return self.xtm.lnpi


    def pad(self, axes=None, ffill=True, bfill=False, limit=None, inplace=False):
        """
        pad nan values in underlying data to values

        Parameters
        ----------
        ffill : bool, default=True
            do forward filling
        bfill : bool, default=False
            do back filling
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.
        inplace : bool, default=False

        Returns
        -------
        out : lnPi
            padded object
        """
        from ._utils import ffill, bfill
        import bottleneck

        if axes is None:
            axes = range(self.ndim)

        data = self.data
        datas = []

        if ffill:
            datas += [ffill(data, axis=axis, limit=limit) for axis in axes]
        if bfill:
            datas += [bfill(data, axis=axis, limit=limit) for axis in axes]

        if len(datas) > 0:
            data = bottleneck.nanmean(datas, axis=0)

        if inplace:
            new = self
            new._clear_cache()
        else:
            new = self.copy()

        new.data[...] = data
        return new

    def zeromax(self, inplace=False):
        """
        shift so that lnpi.max() == 0
        """

        if inplace:
            new = self
            self._clear_cache()
        else:
            new = self.copy()

        new.data[...] = new.data - new.max()
        return new

    def adjust(self, zeromax=False, pad=False, inplace=False):
        """
        do multiple adjustments in one go
        """

        if inplace:
            new = self
        else:
            new = self.copy()

        if zeromax:
            new.zeromax(inplace=True)
        if pad:
            new.pad(inplace=True)
        return new


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

        new = self.copy()
        new._optinfo['mu'] = mu

        dmu = new.mu - self.mu

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

        new.data[...] += shift
        new.adjust(zeromax=zeromax, pad=pad, inplace=True)

        return new


    def smooth(self,
               sigma=4,
               mode='nearest',
               truncate=4,
               inplace=False,
               **kwargs):
        """
        apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool (Default False)
         if True, do inplace modification.
        mode, truncate : arguments to filters.gaussian_filter

        **kwargs : (Default sigma=4, mode='nearest',truncate=4)
         arguments to filters.gaussian_filter
        """

        if inplace:
            new = self
            new._clear_cache()
        else:
            new = self.copy()

        filters.gaussian_filter(
            new.data,
            output=new.data,
            mode=mode,
            truncate=truncate,
            sigma=sigma,
            **kwargs)
        return new



    def copy_shallow(self, mask=None, **kwargs):
        """
        create shallow copy

        Parameters
        ----------
        mask : optional
            if specified, new object has this mask
            otherwise, at least copy old mask
        """
        if mask is None:
            mask = self.mask.copy()

        return self.__class__(
            self.data, mask=mask, fill_value=self.fill_value, **dict(self._optinfo, **kwargs))

    def or_mask(self, mask, **kwargs):
        """
        new object with logical or of self.mask and mask
        """
        return self.copy_shallow(mask=mask + self.mask, **kwargs)

    def and_mask(self, mask, **kwargs):
        """
        new object with logical and of self.mask and mask
        """
        return self.copy_shallow(mask=mask * self.mask, **kwargs)


    @classmethod
    def from_table(cls,
                   path,
                   mu,
                   volume,
                   beta,
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
            **kwargs)




# cached indices
from functools import lru_cache

@lru_cache(maxsize=10)
def _get_indices(shape):
    return np.indices(shape)

@lru_cache(maxsize=10)
def _get_range(n):
    return np.arange(n)


@lru_cache(maxsize=10)
def get_xrlnPiWrapper(shape, n_name='n', mu_name='mu', comp_name='component', phase_name='phase'):
    return xrlnPiWrapper(shape=shape, n_name=n_name, mu_name=mu_name,
                         comp_name=comp_name, phase_name=phase_name)

class xrlnPiWrapper(object):
    """
    this class just wraps lnPi with xarray functionality
    """
    def __init__(self, shape,
                 n_name='n',
                 mu_name='mu',
                 comp_name='component',
                 phase_name='phase',
    ):

        self.shape = shape
        self.ndim  = len(shape)

        # dimension names
        self.n_name = n_name

        self.dims_n = ['{}_{}'.format(n_name, i) for i in range(self.ndim)]
        self.dims_mu = ['{}_{}'.format(mu_name, i) for i in range(self.ndim)]
        self.dims_comp = [comp_name]

    @gcached()
    def coords_n(self):
        return {k:_get_range(n) for k,n in zip(self.dims_n, self.shape)}

    def coords_state(self,mu,**kwargs):
        return dict(zip(self.dims_mu, mu), **kwargs)

    def coords_state_n(self, mu, **kwargs):
        return dict(self.coords_state(mu, **kwargs), **self.coords_n)

    @gcached(prop=False)
    def attrs(self, *args):
        return {
            'dims_n': self.dims_n,
            'dims_mu': self.dims_mu,
            'dims_comp': self.dims_comp,
            'dims_state': self.dims_mu + list(args)
        }


    @gcached(prop=False)
    def ncoords(self, coords_n=False):
        if coords_n:
            coords = self.coords_n
        else:
            coords = None
        return xr.DataArray(
            _get_indices(self.shape),
            dims=self.dims_comp + self.dims_n,
            coords=coords, name=self.n_name)


    def wrap_data(self, data, mu, name='lnPi', coords_n=False, **kwargs):
        if coords_n:
            get_coords = self.coords_state_n
        else:
            get_coords = self.coords_state

        return xr.DataArray(
            data,
            dims=self.dims_n,
            coords=get_coords(mu, **kwargs),
            name=name,
            attrs=self.attrs(*kwargs.keys())
        )

    def wrap_lnpi(self, lnpi, name='lnPi', coords_n=False):
        return self.wrap_data(lnpi.filled(), name=name, coords_n=coords_n, **lnpi._optinfo)


    def wrap_chempot(self, mu, **kwargs):
        return xr.DataArray(mu, dims=self.dims_comp, coords=self.coords_state(mu, **kwargs))




class xrlnPi(object):
    def __init__(self, lnpi_ma, lnpi_wrapper=None):
        self._ma = lnpi_ma
        if lnpi_wrapper is None:
            lnpi_wrapper = get_xrlnPiWrapper(lnpi_ma.shape)
        self._wrapper = lnpi_wrapper

        self._lnpi = self._wrapper.wrap_lnpi(self._ma)


    @property
    def lnpi(self):
        return self._lnpi

    @gcached()
    def chempot(self):
        return self._wrapper.wrap_chempot(self._ma.mu)

    @property
    def ncoords(self):
        return self._wrapper.ncoords()

    @gcached()
    def lnpi_zero(self):
        return self._ma.reshape(-1)[0]

    @gcached()
    def _state(self):
        if self.lnpi.attrs.get('state_as_attrs', 0) != 0:
            return self._lnpi.attrs
        else:
            return self._lnpi.coords
 
    @property
    def volume(self):
        return self._state['volume']

    @property
    def beta(self):
        return self._state['beta']

    @property
    def dims_n(self):
        return self._lnpi.attrs['dims_n']

    @property
    def dims_mu(self):
        return self._lnpi.attrs['dims_mu']

    @property
    def dims_comp(self):
        return self._lnpi.attrs['dims_comp']

    @property
    def dims_state(self):
        return self._lnpi.attrs['dims_state']


    @gcached()
    def coords_state(self):
        return {k: self._lnpi.coords[k].values for k in self.dims_state}

    @gcached(prop=False)
    def max(self):
        return self.lnpi.max(self.dims_n).rename('lnPi_max')

    @gcached(prop=False)
    def argmax_mask(self):
        return self.lnpi == self.max()

    @gcached(prop=False)
    def argmax(self):
        return np.where(self.argmax_mask())

    @gcached()
    def pi(self):
        return np.exp(self.lnpi - self.max()).rename('Pi')

    @gcached()
    def pi_sum(self):
        return self.pi.sum(self.dims_n).rename('Pi_sum')

    @gcached()
    def pi_norm(self):
        return ( self.pi / self.pi_sum ).rename('Pi_norm')


    @gcached()
    def nave(self):
        """average number of particles of each component"""
        return (self.pi_norm * self.ncoords).sum(self.dims_n)

    @property
    def density(self):
        """density of each component"""
        return (self.nave / self.volume).rename('density')

    @gcached()
    def nvar(self):
        return (self.pi_norm * (self.ncoords - self.nave)**2).sum(self.dims_n).rename('nvar')

    @property
    def molfrac(self):
        n = self.nave
        return (n / n.sum(self.dims_comp)).rename('molfrac')

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
            zval = self.lnpi_zero - self.max()
        return (zval - np.log(self.pi_sum)) / self.beta











