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





################################################################################
# xlnPi
# register xtm property to collection classes
from .core import MaskedlnPi, BaselnPiCollection

BaselnPiCollection.register_listaccessor('xtm')
@MaskedlnPi.decorate_accessor('xtm')
class xrlnPi(object):
    """
    accessor for lnpi properties
    """
    def __init__(self, lnpi_ma):
        self._ma = lnpi_ma

        lnpi_wrapper = get_xrlnPiWrapper(lnpi_ma.shape)
        self._wrapper = lnpi_wrapper

        self._lnpi = self._wrapper.wrap_lnpi(self._ma)


    @gcached()
    def lnpi(self):
        return self._wrapper.wrap_lnpi(self._ma)

    @gcached()
    def chempot(self):
        return self._wrapper.wrap_chempot(self._ma.mu)

    @property
    def ncoords(self):
        return self._wrapper.ncoords()

    @gcached()
    def lnpi_zero(self):
        return self._ma.data.reshape(-1)[0]

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

    @gcached()
    def pi(self):
        return self._wrapper.wrap_lnpi(self._ma.pi).rename('Pi')

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
        # Data already available at base level, so use it
        return xr.DataArray(self._ma.omega(zval), coords=self.coords_state)
        # if zval is None:
        #     zval = self.lnpi_zero - self.max()
        # return (zval - np.log(self.pi_sum)) / self.beta




class TMCanonical(object):
    """
    Canonical ensemble properties
    """
    pass



