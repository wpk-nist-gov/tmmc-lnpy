from __future__ import print_function, absolute_import, division

import numpy as np
import xarray as xr
import pandas as pd

from .cached_decorators import gcached
#from . import extensions
#from .core import accessor, listaccessor

from .utils import dim_to_suffix_dataset


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
        return {k:xr.DataArray(_get_range(n), dims=k, attrs={'long_name': r'${}$'.format(k)}) for k,n in zip(self.dims_n, self.shape)}

    def coords_lnz(self, lnz):
        return {k: xr.DataArray(lnz, attrs={'long_name': r'$\beta\mu_{}$'.format(i)} ) for i, (k, lnz) in enumerate(zip(self.dims_lnz, lnz))}

    # def coords_state(self,lnz, **kwargs):
    #     return dict(zip(self.dims_lnz, lnz), **kwargs)
    def coords_state(self, lnz, **kwargs):
        return dict(self.coords_lnz(lnz), **kwargs)

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


    @gcached(prop=False)
    def ncoords_tot(self, coords_n=False):
        return self.ncoords(coords_n).sum(self.dims_comp)

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
                                         cache_list=['betamu', 'nvec', 'dens','betaOmega', 'PE'])
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
    @xr_name(r'$\ln \Pi(n,\mu,V,T)$')
    def lnpi(self):
        return self._wrapper.wrap_lnpi(self._ma, lnz=self._ma.lnz, **self._ma.state_kws)

    @gcached()
    @xr_name(r'$\beta {\bf \mu}$')
    def betamu(self):
        return self._wrapper.wrap_lnz(self._ma.lnz, **self._ma.state_kws)

    @gcached()
    @xr_name(r'$\ln\beta{\bf\mu}$')
    def lnz(self):
        return self._wrapper.wrap_lnz(self._ma.lnz, **self._ma.state_kws)

    @property
    def ncoords(self):
        return self._wrapper.ncoords()


    @property
    def ncoords_tot(self):
        return self._wrapper.ncoords_tot()

    #@gcached()
    # @property
    # def lnpi_zero(self):
    #     return self._ma.data.ravel()[0]

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
        return {k: self.lnpi.coords[k] for k in self.dims_state}


    @gcached()
    def _pi_params(self):
        """
        store pi_norm and pi_sum for later calculations
        """
        lnpi_local_max = self._ma.local_max()
        pi = np.exp(self._ma - lnpi_local_max)
        pi_sum = pi.sum()
        pi_norm =  pi / pi_sum

        lnpi_zero = self._ma.data.ravel()[0] -lnpi_local_max

        # wrap results
        pi_sum = xr.DataArray(pi_sum, coords=self.coords_state,
                              attrs={'long_name': r'$\sum \tilde{\Pi}(n)$'}, name='pi_sum')

        pi_norm = (
            self._wrapper.wrap_lnpi(pi_norm, lnz=self._ma.lnz,
                                    **self._ma.state_kws)
            .rename('pi_norm')
            .assign_attrs(long_name=r'$\Pi(n)$',
                          standard_name='normalized_distribution')
        )
        return pi_sum, pi_norm, lnpi_zero


    @property
    def pi_sum(self):
        return self._pi_params[0]

    @property
    def pi_norm(self):
        return self._pi_params[1]

    @property
    def _lnpi_zero(self):
        return self._pi_params[2]


    @property
    def lnpi_norm(self):
        """
        fill zeros with nan
        """
        pi = self.pi_norm
        return np.log(xr.where(pi > 0, pi, np.nan))

    def mean_pi(self, x, *args, **kwargs):
        """
        sum(Pi * x)

        x can be an array or a callable

        f(self, *args, **kwargs)

        """
        if callable(x):
            x = x(self, *args, **kwargs)
        return (self.pi_norm * x).sum(self.dims_n)

    def var_pi(self, x, y=None, *args, **kwargs):
        """
        given x(N) and y(N), calculate

        v = <x(N) y(N)> - <x(N)> <y(N)>

        x and y can be arrays, or callables, in which case:
        x = x(self, *args, **kwargs)

        etc.

        """

        if callable(x):
            x = x(self, *args, **kwargs)
        x_mean = self.mean_pi(x)

        if y is None:
            y = x
        if y is x:
            y_mean = x_mean
        else:
            if callable(y):
                y = y(self, *args, **kwargs)
            y_mean = self.mean_pi(y)
        return self.mean_pi((x - x_mean) * (y - y_mean))

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    @gcached()
    @xr_name(r'${\bf n}(\mu,V,T)$')
    def nvec(self):
        """average number of particles of each component"""
        return self.mean_pi(self.ncoords)

    @gcached()
    @xr_name(r'$n(\mu,V,T)$')
    def ntot(self):
        return self.mean_pi(self.ncoords_tot)

    @gcached()
    @xr_name(r'$var[{\bf n}(\mu,V,T)]$')
    def nvec_var(self):
        return self.var_pi(self.ncoords)

    @gcached()
    @xr_name(r'$var[n(\mu,V,T)]$')
    def ntot_var(self):
        """variance in total number of particles"""
        return self.var_pi(self.ncoords_tot)

    @property
    @xr_name(r'${\bf \rho}(\mu,V,T)$')
    def dens(self):
        """density of each component"""
        # NOTE: keep this here because of some internal calculations
        return self.nvec.pipe(lambda x: x / x['volume'])

    def argmax(self, *args, **kwargs):
        return self._ma.local_argmax(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self._ma.local_max(*args, **kwargs)

    @xr_name('distance from upper edge')
    def edge_distance(self, ref, *args, **kwargs):
        """distance from endpoint"""
        return xr.DataArray(self._ma.edge_distance(ref, *args, **kwargs), coords=self.coords_state)

    @gcached(prop=False)
    @xr_name(r'$\beta \Omega(\mu,V,T)$', standard_name='grand_potential')
    def betaOmega(self, lnpi_zero=None):
        """
        beta * (Grand potential) = - beta * p * V

        Parameters
        ----------
        lnpi_zero : float or None
         if None, lnpi_zero = self.data.ravel()[0]
        """

        if lnpi_zero is None:
            lnpi_zero = self._lnpi_zero
        return (lnpi_zero - np.log(self.pi_sum))


    @gcached()
    @xr_name(r'${\rm PE}(\mu,V,T)$', standard_name='potential_energy')
    def PE(self):
        """
        beta *(Internal energy)
        """
        # if betaPE available, use that:
        PE = self._ma.extra_kws.get('PE', None)
        if PE is None:
            raise AttributeError('must set "PE" in "extra_kws" of MaskedlnPi')
        return self.mean_pi(PE)

    @xr_name(r'$\beta F(\mu,V,T)$', standard_name='helmholtz_free_energy')
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
@BaselnPiCollection.decorate_accessor('xgce_prop')
@MaskedlnPi.decorate_accessor('xgce_prop')
class xrlnPi_prop(object):
    """
    accessor for extra thermo props
    """
    def __init__(self, x):
        self._x = x
        self._xgce = self._x.xgce

    # when in doubt, return xgce attribute
    # but only consider public attributes
    def __getattr__(self, attr):
        if attr[0] != '_':
            return getattr(self._xgce, attr)
        else:
            raise AttributeError

    @gcached()
    @xr_name(r'$n(\mu,V,T)$', standard_name='total_particles')
    def ntot(self):
        return self._xgce.nvec.pipe(lambda x: x.sum(x.dims_comp, skipna=False))

    @property
    @xr_name(r'${\bf \rho}(\mu,V,T)$')
    def dens(self):
        """density of each component"""
        return self._xgce.nvec.pipe(lambda x: x / x['volume'])

    @property
    @xr_name(r'$\rho(\mu,V,T)$', standard_name='total_density')
    def dens_tot(self):
        return self._xgce.nvec.pipe(lambda x: (x / x['volume']).sum(x.dims_comp, skipna=False))

    @property
    @xr_name(r'${\bf x}(\mu,V,T)$')
    def molfrac(self):
        return self._xgce.nvec / self.ntot

    #@gcached(prop=False)
    @xr_name(r'$\beta\omega(\mu,V,T)', standard_name='grand_potential_per_particle')
    def betaOmega_n(self, lnpi_zero=None):
        """
        beta * (Grand potential) / <n> = -beta * p * V / <n>
        """
        return self._xgce.betaOmega(lnpi_zero) / self.ntot

    #@gcached(prop=False)
    @xr_name(r'$\beta p(\mu,V,T)V$')
    def betapV(self, lnpi_zero=None):
        """
        Note: this is just - beta * Omega
        """
        return -self._xgce.betaOmega(lnpi_zero)


    @property
    @xr_name('mask_stable', description='True where state is most stable')
    def mask_stable(self):
        return self.betapV().pipe(lambda x: x.max('phase') == x)

    #@gcached(prop=False)
    @xr_name(r'$\beta p(\mu,V,T)/\rho$', standard_name='compressibility_factor')
    def Z(self, lnpi_zero=None):
        """
        compressibility factor = beta * P / rho
        """
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r'$p(\mu,V,T)$')
    def pressure(self, lnpi_zero=None):
        return self.betapV(lnpi_zero).pipe(lambda x: x / (x['beta'] * x['volume']))

    @property
    @xr_name(r'${\rm PE}(\mu,V,T)/n$', standard_name='potential_energy_per_particle')
    def PE_n(self):
        """
        beta * (Internal energy) / n
        """
        return self._xgce.PE / self.ntot


    def table(self, keys=None,
              default_keys=['nvec', 'betapV', 'PE_n'],
              edge_dist_ref=None,
              mask_stable=False,
              dim_to_suffix=None,
    ):

        out = []
        if edge_dist_ref is not None:
            out.append(self._xgce.edge_distance(edge_dist_ref))

        if keys is None:
            keys = []
        keys = keys + default_keys

        for key in keys:
            try:
                v = getattr(self, key)
                if callable(v):
                    v = v()
                out.append(v)
            except:
                pass

        out = xr.merge(out)
        if 'lnz' in keys:
            # if including property lnz, then drop lnz_0, lnz_1,...
            out = out.drop(out['lnz']['dims_lnz'])

        if mask_stable:
            # mask_stable inserts nan in non-stable
            mask_stable = self.mask_stable
            phase = out.phase
            out = (
                out
                .where(mask_stable)
                .max('phase')
                .assign(phase=lambda x: phase[mask_stable.argmax('phase')])
            )

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                out = out.pipe(dim_to_suffix_dataset, dim=dim)
        return out

    @property
    @xr_name(r'$\beta G(\mu,V,T)$', standard_name='Gibbs_free_energy')
    def betaG(self):
        """
        beta * (Gibbs free energy)
        """
        return self._xgce.betamu.pipe(lambda betamu: (betamu * self._xgce.nvec).sum(betamu.dims_comp))

    @property
    @xr_name(r'$\beta G(\mu,V,T)/n$', standard_name='Gibbs_free_energy_per_particle')
    def betaG_n(self):
        """
        beta * (Gibbs free energy) / n
        """
        return self.betaG / self.ntot

    @xr_name(r'$\beta F(\mu,V,T)$', standard_name='helmholtz_free_energy')
    def betaF(self, lnpi_zero=None):
        """
        beta * (Helmholtz free energy)
        """
        return self._xgce.betaOmega(lnpi_zero) + self.betaG

    @xr_name(r'$\beta F(\mu,V,T)/n$', standard_name='helmholtz_free_energy_per_particle')
    def betaF_n(self, lnpi_zero=None):
        """
        beta * (Helmholtz free energy) / n
        """
        return self.betaF(lnpi_zero) / self.ntot

    @xr_name(r'$\beta E(\mu,V,T)$', standard_name='total_energy')
    def betaE(self, ndim=3):
        """
        Total energy
        """
        return (
            ndim/2. * self.ntot +
            self._xgce.PE.pipe(lambda x: x * x.beta)
        )

    @xr_name(r'$\beta E(\mu,V,T)/ n$', standard_name='total_energy_per_particle')
    def betaE_n(self, ndim=3):
        """
        (Total energy) / n
        """
        return self.betaE(ndim) / self.ntot

    @xr_name(r'$S(\mu,V,T) / k_{\rm B}$', standard_name='entropy')
    def S(self, lnpi_zero=None, ndim=3):
        """
        Entropy / kB
        """
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r'$S(\mu,V,T)/(n kB)$', standard_name='entropy_per_particle')
    def S_n(self, lnpi_zero=None, ndim=3):
        """
        Entropy / (N kB)
        """
        return self.S(lnpi_zero, ndim) / self.ntot




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
            # NOTE: if don't use '.values', then get extra coords don't want
            .where(~self._xgce.lnpi.isnull().values)
        )


    @property
    def nvec(self):
        return self.ncoords

    @gcached()
    def ntot(self):
        return self.ncoords.sum(self._xgce.dims_comp)


    @gcached(prop=False)
    @xr_name(r'$\beta F({\bf n},V,T)$', standard_name='helmholtz_free_energy')
    def betaF(self, lnpi_zero=None):
        """
        Helmholtz free energy
        """
        x= self._ma.xgce

        if lnpi_zero is None:
            lnpi_zero = x.lnpi_zero

        return (
            (-(x.lnpi - lnpi_zero) +
            (x.ncoords * x.betamu).sum(x.dims_comp))
            .assign_coords(**x._wrapper.coords_n)
            .drop(x.dims_lnz)
            .assign_attrs(x._standard_attrs)
        )

    @xr_name(r'$\beta F({\bf n},V,T)/n$', standard_name='helmholtz_free_energy_per_particle')
    def betaF_n(self, lnpi_zero=None):
        return self.betaF(lnpi_zero) / self.ntot

    @gcached()
    @xr_name(r'${\rm PE}({\bf n},V,T)/n$', standard_name='potential_energy')
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

    @property
    @xr_name(r'${\rm PE}({\bf n},V,T)/n$', standard_name='potential_energy_per_particle')
    def PE_n(self):
        return self.PE / self.ntot

    @xr_name(r'$\beta E({\bf n},V,T)$')
    def betaE(self, ndim=3):
        return ndim / 2 * self.ntot + self._xgce.beta * self.PE

    @xr_name(r'$\beta E({\bf n},V,T)/n$')
    def betaE_n(self, ndim=3):
        return self.betaE(ndim) / self.ntot

    @xr_name(r'$S({\bf n},V,T)/k_{\rm B}')
    def S(self, ndim=3, lnpi_zero=None):
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r'$S({\bf n},V,T)/(n k_{rm B})$')
    def S_n(self, ndim=3, lnpi_zero=None):
        return self.S(ndim, lnpi_zero) / self.ntot

    @gcached(prop=False)
    @xr_name(r'$\beta {\bf\mu}({bf n},V,T)$', standard_name='absolute_activity')
    def betamu(self, lnpi_zero=None):
        """Canonical beta*(chemial potential)"""
        return (
            xr.concat([self.betaF(lnpi_zero).differentiate(n) for n in self._xgce.dims_n],
                      dim=self._xgce.dims_comp[0])
            .assign_attrs(self._xgce._standard_attrs)
        )

    @property
    @xr_name(r'${\bf \rho}({\bf n},V,T)$')
    def dens(self):
        return self.ncoords / self._xgce.volume

    @gcached(prop=False)
    @xr_name(r'$\beta\Omega({\bf n},V,T)$')
    def betaOmega(self, lnpi_zero=None):
        """
        beta * Omega = betaF - lnz .dot. N
        """
        return self.betaF(lnpi_zero) - (self.betamu(lnpi_zero) * self.ncoords).sum(self._xgce.dims_comp)

    @xr_name(r'$\beta\Omega({\bf n},V,T)$')
    def betaOmega_n(self, lnpi_zero=None):
        return self.betaOmega(lnpi_zero) / self.ntot

    @xr_name(r'$\beta p({\bf n},V,T)V$')
    def betapV(self, lnpi_zero=None):
        """
        beta * p * V = - beta * Omega
        """
        return - self.betaOmega(lnpi_zero)

    @xr_name(r'$\beta p({\bf n},V,T)/\rho}')
    def Z(self, lnpi_zero=None):
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r'$p({\bf n},V,T)$')
    def pressure(self):
        return self.betapV.pipe(lambda x: x / (x.beta * x.volume))

    def table(self, keys=None, default_keys=['betamu','betapV','PE_n','betaF_n'], dim_to_suffix=None):
        out = []
        if keys is None:
            keys = []

        for key in keys + default_keys:
            try:
                v = getattr(self, key)
                if callable(v):
                    v = v()
                out.append(v)
            except:
                pass

        out = xr.merge(out)

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                out = out.pipe(dim_to_suffix_dataset, dim=dim)
        return out








