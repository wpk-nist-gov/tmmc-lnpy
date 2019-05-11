import numpy as np
import xarray as xr

from functools import wraps, partial
from lnPi.cached_decorators import cached_clear, cached, cached_func, gcached


from scipy.ndimage import filters


# NOTE: this is a work in progress
# I like the idea of having an xarray
# accessor, but it might be more trouble than
# it's worth

# NOTE : I messed around wtih a bunch
# of stuff, and the old way is probably
# best.  it is very fast to
# construct xarray objects
# so just do that


# class lnPiwrapper(object):
#     """
#     log(Pi) object

#     """
#     def __init__(self, lnpi, **attrs):
#         self._lnpi = lnpi
#         if len(attrs) > 0:
#             self._lnpi.attrs.update(**attrs)

#     @property
#     def tm(self):
#         """
#         accessort to transity matrix stuff
#         """
#         return self._lnpi.tm

#     @property
#     def tmgc(self):
#         """
#         access to grand canonical calculations
#         """


#     @classmethod
#     def from_data(self):
#         pass


# cached indices
_INDICES = {}
def _get_indices(shape):
    if shape not in _INDICES:
        _INDICES[shape] = np.indices(shape)
    return _INDICES[shape]


@xr.register_dataarray_accessor('tm')
class TM(object):

    def __init__(self, da):
        self._da = da
        self._cache = {}

    @gcached()
    def _state(self):
        if self._da.attrs.get('state_as_attrs', 0) != 0:
            return self._da.attrs
        else:
            return self._da.coords
    @property
    def volume(self):
        return self._state['volume']

    @property
    def beta(self):
        return self._state['beta']

    @property
    def dims_n(self):
        return self._da.attrs['dims_n']

    @property
    def dim_mu(self):
        return self._da.attrs['dims_mu']

    @property
    def dims_comp(self):
        return self._da.attrs['dims_comp']

    @property
    def dims_state(self):
        return self._da.attrs['dims_state']

    @property
    def lnpi_zero(self):
        return self._da.attrs['lnpi_zero']

    @gcached()
    def coords_state(self):
        return {k: self._da.coords[k].values for k in self.dims_state}

    @property
    def ncomp(self):
        return len(self.dims_n)

    @property
    def lnpi(self):
        return self._da

    @gcached()
    def lnpi_max(self):
        return self.lnpi.max(self.dims_n).rename('lnpi_max')

    @gcached()
    def argmax_mask(self):
        return self.lnpi == self.lnpi_max

    @gcached
    def argmax(self):
        return np.where(self.argmax_mask)

    @gcached()
    def chempot(self):
        mus = [self._state[k] for k in self._da.attrs['dims_mu']]
        return xr.concat(mus, dim=self.dims_comp[0])

    @gcached()
    def ncoords(self):
        """particle number for each particle dimension"""
        shape = tuple(self.lnpi.sizes[k] for k in self.dims_n)
        return xr.DataArray(_get_indices(shape), dims=self.dims_comp + self.dims_n)


    def reweight(self, mu, zero_max=False):
        mu = np.atleast_1d(mu)
        if len(mu) != self.ncomp:
            raise ValueError('len(mu) != {}'.format(self.ncomp))
        new = self.lnpi + (self.ncoords * self.chempot).sum(self.dims_comp)

        if zero_max:
            # shift output
            lnpi_max = new.lnpi.max()
            new.lnpi -= lnpi_max

        # copy over attributes
        new.attrs.update(**self.lnpi.attrs)
        # update lnpi_zero
        new.attrs['lnpi_zero'] = new.values.reshape(-1)[0]

        # update mu state
        return new.assign_coords(**dict(zip(self.dims_mu, mu)))


    def ffill(self, dims=None):
        """
        ffill the lnPi array across dims

        Parameters
        ----------
        dims : optional, default=`self.dims_n`
        dimensions to fill over.  Each is taken in turn, then averaged
        """

        if dims is None:
            dims = self.dims_n
        return sum(self.lnpi.ffill(x) for x in dims) / len(dims)


    def bfill(self, dims=None):
        """
        bfill the lnPi array across dims

        Parameters
        ----------
        dims : optional, default=`self.dims_n`
        dimensions to fill over.  Each is taken in turn, then averaged
        """

        if dims is None:
            dims = self.dims_n
        return sum(self.lnpi.bfill(x) for x in dims) / len(dims)


    def smooth(self, sigma=4, mode='nearest', truncate=4, ffill=False, bfill=False, **kwargs):
        """
        apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool (Default False)
         if True, do inplace modification.


        **kwargs : (Default sigma=4, mode='nearest',truncate=4)
         arguments to filters.gaussian_filter
        """

        if not bfill and not ffill:
            lnpi = self.lnpi.copy()
            mask = None
        else:
            # save the mask for later
            mask = self.lnpi.isnull()
            lnpi = self.lnpi

            if ffill:
                lnpi = lnpi.tm.ffill()
            if bfill:
                lnpi = lnpi.tm.bfill()

        filters.gaussian_filter(
            lnpi.values, output=lnpi.values,
            mode=mode, truncate=truncate, sigma=sigma, **kwargs)

        if mask is not None:
            lnpi.values[mask] = np.nan
        return lnpi









@xr.register_dataarray_accessor('tmgc')
class TMGC(object):
    def __init__(self, da):
        self._da = da
        self._cache = {}
        self.tm = self._da.tm


    @gcached()
    def pi(self):
        return np.exp(self.tm.lnpi - self.tm.lnpi_max).rename('pi')

    @gcached()
    def pi_sum(self):
        return self.pi.sum(self.tm.dims_n).rename('pi_sum')

    @gcached()
    def pi_norm(self):
        return ( self.pi / self.pi_sum ).rename('pi_norm')


    @gcached()
    def nave(self):
        """average number of particles of each component"""
        return (self.pi_norm * self.tm.ncoords).sum(self.tm.dims_n)

    @property
    def density(self):
        """density of each component"""
        return (self.nave / self.tm.volume).rename('density')

    @gcached()
    def nvar(self):
        return (self.pi_norm * (self.tm.ncoords - self.nave)**2).sum(self.tm.dims_n).rename('nvar')

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
            zval = self.tm.lnpi_zero - self.tm.lnpi_max
        return (zval - np.log(self.pi_sum)) / self.tm.beta



