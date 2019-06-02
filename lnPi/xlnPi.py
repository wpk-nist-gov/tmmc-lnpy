from __future__ import print_function, absolute_import, division

import numpy as np
import xarray as xr
import pandas as pd

from .cached_decorators import gcached
#from . import extensions
#from .core import accessor, listaccessor


################################################################################
# xlnPi wrapper
from functools import lru_cache

@lru_cache(maxsize=10)
def _get_indices(shape):
    return np.indices(shape)

@lru_cache(maxsize=10)
def _get_range(n):
    return np.arange(n)


@lru_cache(maxsize=10)
def get_xrlnPiWrapper(shape, n_name='n', mu_name='mu', comp_name='component',
                      phase_name='phase'):
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

    def wrap_data(self, data, mu, coords_n=False, **kwargs):
        if coords_n:
            get_coords = self.coords_state_n
        else:
            get_coords = self.coords_state

        return xr.DataArray(
            data,
            dims=self.dims_n,
            coords=get_coords(mu, **kwargs),
            attrs=self.attrs(*kwargs.keys())
        )

    def wrap_lnpi(self, lnpi, mu, coords_n=False, **kwargs):
        return self.wrap_data(lnpi.filled(), mu=mu, coords_n=coords_n, **kwargs)

    def wrap_chempot(self, mu, **kwargs):
        return xr.DataArray(mu, dims=self.dims_comp, coords=self.coords_state(mu, **kwargs))



################################################################################
# xlnPi
# register xgce property to collection classes
from .core import MaskedlnPi, BaselnPiCollection
from functools import wraps
def xr_name(long_name=None, name=None, **kwargs):
    """
    decorator to add name, longname to xarray output
    """
    def decorator(func):
        if name is None:
            _name = func.__name__
        else:
            _name = name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs).rename(_name)
            if long_name is not None:
                out = out.assign_attrs(long_name=long_name, *kwargs)
            return out
        return wrapper
    return decorator




BaselnPiCollection.register_listaccessor('xgce')
@MaskedlnPi.decorate_accessor('xgce')
class xrlnPi(object):
    """
    accessor for lnpi properties
    """
    def __init__(self, lnpi_ma):
        self._ma = lnpi_ma

        lnpi_wrapper = get_xrlnPiWrapper(lnpi_ma.shape)
        self._wrapper = lnpi_wrapper


    @gcached()
    def _standard_attrs(self):
        return self._wrapper.attrs(*self._ma.state_kws.keys())

    #@gcached() to much memory
    @property
    @xr_name('ln[Pi(n; mu,V,T)]')
    def lnpi(self):
        return self._wrapper.wrap_lnpi(self._ma, mu=self._ma.mu, **self._ma.state_kws)

    @gcached()
    @xr_name()
    def mu(self):
        return self._wrapper.wrap_chempot(self._ma.mu, **self._ma.state_kws)

    @property
    def ncoords(self):
        return self._wrapper.ncoords()

    #@gcached()
    @property
    def lnpi_zero(self):
        return self._ma.data.reshape(-1)[0]

    @property
    def volume(self):
        return self._ma.volume
    @property
    def beta(self):
        return self._ma.beta

    @property
    def dims_n(self):
        return self._wrapper.dims_n
    @property
    def dims_mu(self):
        return self._wrapper.dims_mu
    @property
    def dims_comp(self):
        return self._wrapper.dims_comp
    @property
    def dims_state(self):
        return self.lnpi.attrs['dims_state']
    @gcached()
    def coords_state(self):
        return {k: self.lnpi.coords[k].values for k in self.dims_state}

    #@gcached()
    @property
    @xr_name('Pi(n; mu,V,T)')
    def pi(self):
        return self._wrapper.wrap_lnpi(self._ma.pi, mu=self._ma.mu, **self._ma.state_kws)
    @property
    @xr_name('sum_{n} Pi(n; mu,V,T)')
    def pi_sum(self):
        return xr.DataArray(self._ma.pi_sum, coords=self.coords_state)
    #@gcached()
    @property
    @xr_name('normalized Pi')
    def pi_norm(self):
        return ( self.pi / self.pi_sum )

    @gcached()
    @xr_name('<n(component,mu,V,T)>')
    def nave(self):
        """average number of particles of each component"""
        return (self.pi_norm * self.ncoords).sum(self.dims_n)
    @gcached()
    @xr_name('var[n(component,mu,V,T)]')
    def nvar(self):
        return (self.pi_norm * (self.ncoords - self.nave)**2).sum(self.dims_n)
    @property
    @xr_name('density(component,mu,V,T)')
    def density(self):
        """density of each component"""
        return self.nave / self.volume
    @property
    @xr_name('mole_fraction(component,mu,V,T)')
    def molfrac(self):
        n = self.nave
        return n / n.sum(self.dims_comp)

    def argmax(self, *args, **kwargs):
        return self._ma.local_argmax(*args, **kwargs)
    def max(self, *args, **kwargs):
        return self._ma.local_max(*args, **kwargs)

    @xr_name('distance from upper edge(s)')
    def edge_distance(self, ref, *args, **kwargs):
        """distance from endpoint"""
        return xr.DataArray(self._ma.edge_distance(ref, *args, **kwargs), coords=self.coords_state)

    @gcached(prop=False)
    @xr_name('-p(mu,V,T) V')
    def omega(self, zval=None):
        """
        get omega = zval - ln(sum(pi))

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """
        return xr.DataArray(self._ma.omega(zval), coords=self.coords_state)
    @gcached(prop=False)
    @xr_name('pressure(mu,V,T)')
    def pressure(self, zval=None):
        return -self.omega(zval) / self.volume

    @gcached()
    @xr_name('PE(mu,V,T)', standard_name='Potential energy')
    def pe(self):
        pe_N = self._ma.extra_kws.get('pe',None)
        if pe_N is None:
            raise AttributeError('must set "pe" in "extra_kws" of MaskedlnPi')
        return (self.pi_norm * pe_N).sum(self.dims_n)
    @gcached()
    @xr_name('PE(mu,V,T)/N(mu,V,T)', standard_name='Potential energy per particle')
    def pe_n(self):
        return self.pe / self.nave.sum(self.dims_comp)


@MaskedlnPi.decorate_accessor('xcan')
class TMCanonical(object):
    """
    Canonical ensemble properties
    """

    def __init__(self, ma):
        self._ma = ma
        self._xgce = ma.xgce

    @gcached()
    def ncoords(self):
        return (
            self._xgce.ncoords
            .where(~self._xgce.lnpi.isnull())
        )

    @gcached()
    @xr_name('F(n,V,T)', standard_name='Helmholtz free energy')
    def f(self):
        """
        Helmholtz free energy
        """
        x= self._ma.xgce

        return (
            (-(x.lnpi - x.lnpi_zero) / x.beta +
            (x.ncoords * x.mu).sum(x.dims_comp))
            .assign_coords(**x._wrapper.coords_n)
            .drop(x.dims_mu)
            .assign_attrs(x._standard_attrs)
        )


    @gcached()
    @xr_name('PE(n,V,T)')
    def pe(self):
        pe_N = self._ma.extra_kws.get('pe', None)
        if pe_N is None:
            raise AttributeError('must set "pe" in "extra_kws" of MaskedlnPi')

        x = self._xgce
        coords = dict(x._wrapper.coords_n, **self._ma.state_kws)
        return xr.DataArray(
            pe_N,
            dims=x.dims_n,
            coords=coords,
            attrs = x._standard_attrs)
    @gcached(prop=False)
    @xr_name('KE(n,V,T)')
    def ke(self, ndim=3):
        return (
            ndim / 2. *
            self.ncoords.sum(self._xgce.dims_comp) / self._ma.beta
        )
    @gcached(prop=False)
    @xr_name('E(n,V,T)')
    def energy(self, ndim=3):
        return self.pe + self.ke(ndim)


    @gcached()
    @xr_name('mu(component,n,V,T)')
    def mu(self):
        """Canonical chemial potential"""
        return (
            xr.concat([self.F.differentiate(n) for n in self._xgce.dims_n],
                      dim=self._xgce.dims_comp[0])
            .assign_attrs(self._xgce._standard_attrs)
        )
    @gcached()
    @xr_name('density(component,n,V,T)')
    def density(self):
        return self.ncoords / self._xgce.volume

    @gcached(prop=False)
    @xr_name('pressure(n,V,T)')
    def pressure(self):
        """
        Canonical pressure, P = -1/V (F - mu .dot. N)
        """
        P = - 1.0 / self._xgce.volume * (
            self.F -
            (self.mu * self.ncoords).sum(self._xgce.dims_comp))
        return P





