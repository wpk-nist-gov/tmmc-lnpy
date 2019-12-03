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
def get_xrlnPiWrapper(shape, n_name='n', lnz_name='lnz', comp_name='component',
                      phase_name='phase'):
    return xrlnPiWrapper(shape=shape, n_name=n_name, lnz_name=lnz_name,
                         comp_name=comp_name, phase_name=phase_name)

class xrlnPiWrapper(object):
    """
    this class just wraps lnPi with xarray functionality
    """
    def __init__(self, shape,
                 n_name='n',
                 lnz_name='lnz',
                 comp_name='component',
                 phase_name='phase',
    ):

        self.shape = shape
        self.ndim  = len(shape)

        # dimension names
        self.n_name = n_name

        self.dims_n = ['{}_{}'.format(n_name, i) for i in range(self.ndim)]
        self.dims_lnz = ['{}_{}'.format(lnz_name, i) for i in range(self.ndim)]
        self.dims_comp = [comp_name]

    @gcached()
    def coords_n(self):
        return {k:_get_range(n) for k,n in zip(self.dims_n, self.shape)}

    def coords_state(self,lnz,**kwargs):
        return dict(zip(self.dims_lnz, lnz), **kwargs)

    def coords_state_n(self, lnz, **kwargs):
        return dict(self.coords_state(lnz, **kwargs), **self.coords_n)

    @gcached(prop=False)
    def attrs(self, *args):
        return {
            'dims_n': self.dims_n,
            'dims_lnz': self.dims_lnz,
            'dims_comp': self.dims_comp,
            'dims_state': self.dims_lnz + list(args)
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

    def wrap_data(self, data, lnz, coords_n=False, **kwargs):
        if coords_n:
            get_coords = self.coords_state_n
        else:
            get_coords = self.coords_state

        return xr.DataArray(
            data,
            dims=self.dims_n,
            coords=get_coords(lnz, **kwargs),
            attrs=self.attrs(*kwargs.keys())
        )

    def wrap_lnpi(self, lnpi, lnz, coords_n=False, **kwargs):
        return self.wrap_data(lnpi.filled(), lnz=lnz, coords_n=coords_n, **kwargs)

    def wrap_lnz(self, lnz, **kwargs):
        return xr.DataArray(lnz, dims=self.dims_comp, coords=self.coords_state(lnz, **kwargs))



################################################################################
# xlnPi
# register xgce property to collection classes
from .core import MaskedlnPi, BaselnPiCollection
from functools import wraps
def xr_name(long_name=None, name=None, **kws):
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
            if hasattr(self, '_standard_attrs'):
                attrs = self._standard_attrs
            else:
                attrs = {}
            attrs = dict(attrs, **kws)

            if long_name is not None:
                attrs['long_name'] = long_name
            out = out.assign_attrs(**attrs)
            return out
        return wrapper
    return decorator




BaselnPiCollection.register_listaccessor('xgce',
                                         cache_list=['density', 'nave', 'nvar', 'molfrac', 'betaOmega', 'Z'])
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
    @xr_name('ln[Pi(n; lnz,V,T)]')
    def lnpi(self):
        return self._wrapper.wrap_lnpi(self._ma, lnz=self._ma.lnz, **self._ma.state_kws)

    @gcached()
    @xr_name()
    def lnz(self):
        return self._wrapper.wrap_lnz(self._ma.lnz, **self._ma.state_kws)

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
    def dims_lnz(self):
        return self._wrapper.dims_lnz
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
    @xr_name('Pi(n; lnz,V,T)')
    def pi(self):
        return self._wrapper.wrap_lnpi(self._ma.pi, lnz=self._ma.lnz, **self._ma.state_kws)

    @property
    @xr_name('sum_{n} Pi(n; lnz,V,T)')
    def pi_sum(self):
        return xr.DataArray(self._ma.pi_sum, coords=self.coords_state)

    #@gcached()
    @property
    @xr_name('normalized Pi')
    def pi_norm(self):
        return ( self.pi / self.pi_sum )

    @property
    def lnpi_norm(self):
        """
        fill zeros with nan
        """
        pi = self.pi_norm
        return np.log(xr.where(pi > 0, pi, np.nan))


    def mean_pi(self, x):
        """
        sum(Pi * x)
        """

        return (self.pi_norm * x).sum(self.dims_n)

    def var_pi(self, x, y=None):
        """
        given x(N) and y(N), calculate

        v = <x(N) y(N)> - <x(N)> <y(N)>
        """

        x_mean = self.mean_pi(x)

        if y is None:
            y = x
        if y is x:
            y_mean = x_mean
        else:
            y_mean = self.mean_pi(y)

        return self.mean_pi((x - x_mean) * (y - y_mean))


    @gcached(prop=False)
    @xr_name('<n(component,lnz,V,beta)>')
    def nave(self):
        """average number of particles of each component"""
        return self.mean_pi(self.ncoords)

    @gcached(prop=False)
    @xr_name('<n(lnz, V, beta)>')
    def ntot(self):
        return self.nave().sum(self.dims_comp)

    @gcached(prop=False)
    @xr_name('var[n(component,lnz,V,beta)]')
    def nvar(self):
        return self.var_pi(self.ncoords)

    @xr_name('density(component,lnz,V,beta)')
    def density(self):
        """density of each component"""
        return self.nave() / self.volume

    @xr_name('rho(component, lnz, V, beta)')
    def rho(self):
        """
        density of each component
        """
        return self.density()

    @xr_name('rho(lnz, V, beta)')
    def rho_tot(self):
        return self.rho().sum(self.dims_comp)


    @xr_name('mole_fraction(component,lnz,V,beta)')
    def molfrac(self):
        return self.nave() / self.ntot()

    def argmax(self, *args, **kwargs):
        return self._ma.local_argmax(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self._ma.local_max(*args, **kwargs)

    @xr_name('distance from upper edge(s)')
    def edge_distance(self, ref, *args, **kwargs):
        """distance from endpoint"""
        return xr.DataArray(self._ma.edge_distance(ref, *args, **kwargs), coords=self.coords_state)

    @gcached(prop=False)
    @xr_name('beta * Omega(lnz,V,beta)')
    def betaOmega(self, zval=None):
        """
        beta * (Grand potential) = - beta * p * V

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """
        return xr.DataArray(self._ma.betaOmega(zval), coords=self.coords_state)

    @gcached(prop=False)
    @xr_name('beta * Omega(lnz, V, beta) / n')
    def betaOmega_n(self, zval=None):
        """
        beta * (Grand potential) / <n> = -beta * p * V / <n>
        """
        return self.betaOmega(zval) / self.ntot()

    @gcached(prop=False)
    @xr_name('beta * p(lnz,V,beta) * V')
    def betapV(self, zval=None):
        return -self.betaOmega(zval)

    @gcached(prop=False)
    @xr_name('beta * p(lnZ, V, beta) V / <n(lnz, V, beta)>')
    def Z(self, zval=None):
        """
        compressibility factor = beta * P / rho
        """
        return -self.betaOmega_n(zval)

    @gcached(prop=False)
    @xr_name('PE(lnz,V,beta)', standard_name='Potential energy')
    def PE(self):
        """
        beta *(Internal energy)
        """
        # if betaPE available, use that:
        PE = self._ma.extra_kws.get('PE', None)
        if PE is None:
            raise AttributeError('must set "PE" in "extra_kws" of MaskedlnPi')
        return self.mean_pi(PE)

    @xr_name('beta * PE(lnz, V, beta)/n')
    def PE_n(self):
        """
        beta * (Internal energy) / n
        """
        return self.PE() / self.ntot()

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)


    def betaF_alt(self, betaF_can, correction=True):
        """
        calculate betaF from formula

        betaF = sum(Pi(N) * betaF(N))

        if correction is True, then include:
        betaF = sum(Pi * (betaF(N) + lnPi(N)))
        """
        if correction:
            betaF_can = betaF_can + self.lnpi_norm
        return (self.pi_norm * betaF_can).sum(self.dims_n)




# this would act on individual pis
#BaselnPiCollection.register_listaccessor('xgce_fe',cache_list=[])
# this acts on collective
@BaselnPiCollection.decorate_accessor('xgce_fe')
@MaskedlnPi.decorate_accessor('xgce_fe')
class xrlnPi_extras(object):
    """
    accessor for extra thermo props
    """
    def __init__(self, x):
        self._x = x
        self._xgce = self._x.xgce

    @gcached(prop=False)
    @xr_name('beta * G(lnz, V, beta)')
    def betaG(self):
        """
        beta * (Gibbs free energy)
        """
        lnz = self._xgce.lnz
        nave = self._xgce.nave()
        return (lnz * nave).sum(lnz.dims_comp)

    @xr_name('beta * G(lnz, V, beta)/<n>')
    def betaG_n(self):
        """
        beta * (Gibbs free energy) / n
        """
        return self.betag() / self._xgce.ntot()

    @gcached(prop=False)
    @xr_name('beta * F(lnz, V, beta)')
    def betaF(self, zval=None):
        """
        beta * (Helmholtz free energy)
        """
        return self._xgce.betaOmega(zval) + self.betaG()

    @xr_name('beta * F(lnz, V, beta) / n')
    def betaF_n(self, zval=None):
        """
        beta * (Helmholtz free energy) / n
        """
        return self.betaF(zval) / self._xgce.ntot()

    @gcached(prop=False)
    @xr_name('beta * E(lnz, V, beta)', standard_name='total energy')
    def betaE(self, ndim=3):
        """
        Total energy
        """
        return (
            ndim/2. * self._xgce.ntot() +
            self._xgce.PE().pipe(lambda x: x * x.beta)
        )

    @xr_name('beta * U(lnz, V, beta)/<n>')
    def betaE_n(self, ndim=3):
        """
        (Total energy) / n
        """
        return self.betaE(ndim) / self._xgce.ntot()

    @gcached(prop=False)
    @xr_name('S(lnz, V, beta)/kB', standard_name='entropy')
    def S(self, zval=None, ndim=3):
        """
        Entropy / kB
        """
        return self.betaE(ndim) - self.betaF(zval)

    @xr_name('S(lnz, V, beta)/(N kB)')
    def S_n(self, zval=None, ndim=3):
        """
        Entropy / (N kB)
        """
        return self.S(zval, ndim) / self._xgce.ntot()




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


    @gcached(prop=False)
    def ntot(self):
        return self.ncoords.sum(self._xgce.dims_comp)

    @gcached(prop=False)
    @xr_name('beta * F(n,V,T)', standard_name='Helmholtz free energy')
    def betaF(self):
        """
        Helmholtz free energy
        """
        x= self._ma.xgce

        return (
            (-(x.lnpi - x.lnpi_zero) +
            (x.ncoords * x.lnz).sum(x.dims_comp))
            .assign_coords(**x._wrapper.coords_n)
            .drop(x.dims_lnz)
            .assign_attrs(x._standard_attrs)
        )

    @xr_name('beta * F(n, V, T) / n', standard_name='Helmholtz free energy / N')
    def betaF_n(self):
        return self.betaF() / self.ntot()

    @gcached(prop=False)
    @xr_name('PE(n,V,T)')
    def PE(self):
        # if betaPE available, use that:
        PE = self._ma.extra_kws.get('PE', None)
        if PE is None:
            raise AttributeError('must set "PE" in "extra_kws" of MaskedlnPi')
        x = self._xgce
        coords = dict(x._wrapper.coords_n, **self._ma.state_kws)
        return xr.DataArray(
            PE,
            dims=x.dims_n,
            coords=coords,
            attrs = x._standard_attrs)

    @xr_name('PE(n,V,T)/n')
    def PE_n(self):
        return self.PE() / self.ntot()

    @gcached(prop=False)
    @xr_name('E(n,V,T)')
    def betaE(self, ndim=3):
        return ndim / 2 * self.ntot() + self._xgce.beta * self.PE()

    @xr_name('E(n,V,T)/n')
    def betake(self, ndim=3):
        return self.betaE(ndim) / self.ntot()

    @gcached(prop=False)
    @xr_name('lnz(component,n,V,T)')
    def lnz(self):
        """Canonical beta*(chemial potential)"""
        return (
            xr.concat([self.betaF().differentiate(n) for n in self._xgce.dims_n],
                      dim=self._xgce.dims_comp[0])
            .assign_attrs(self._xgce._standard_attrs)
        )

    @gcached(prop=False)
    @xr_name('density(component,n,V,T)')
    def density(self):
        return self.ncoords / self._xgce.volume

    @xr_name('rho(component, n, V, T)')
    def rho(self):
        return self.density()


    @gcached(prop=False)
    @xr_name('beta * p(n,V,T) * V')
    def betapV(self):
        """ 
        beta * P * V = lnz .dot. N - beta * F
        """
        return (self.lnz() * self.ncoords).sum(self._xgce.dims_comp) - self.betaF()


    @xr_name('beta * P(n,V,T) / rho')
    def Z(self):
        return self.betaPV / self.ntot()






