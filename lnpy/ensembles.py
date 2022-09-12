from __future__ import absolute_import, division, print_function

from functools import lru_cache, wraps

import numpy as np

# import pandas as pd
import xarray as xr

from .cached_decorators import gcached_use_cache as gcached
from .lnpidata import MaskedlnPiDelayed, lnPiMasked
from .lnpiseries import lnPiCollection
from .maskedlnpi_legacy import MaskedlnPiLegacy
from .utils import dim_to_suffix_dataset


###############################################################################
# xlnPi wrapper
@lru_cache(maxsize=10)
def _get_indices(shape):
    return np.indices(shape)


@lru_cache(maxsize=10)
def _get_range(n):
    return np.arange(n)


@lru_cache(maxsize=10)
def get_xrlnPiWrapper(
    shape,
    rec_name="sample",
    n_name="n",
    lnz_name="lnz",
    comp_name="component",
    phase_name="phase",
):
    return xlnPiWrapper(
        shape=shape,
        rec_name=rec_name,
        n_name=n_name,
        lnz_name=lnz_name,
        comp_name=comp_name,
        phase_name=phase_name,
    )


class xlnPiWrapper:
    _use_cache = True
    """
    This class just wraps lnPi with xarray functionality

    Most likely, this shouldn't be accessed by the user.


    """

    def __init__(
        self,
        shape,
        rec_name="sample",
        n_name="n",
        lnz_name="lnz",
        comp_name="component",
        phase_name="phase",
    ):

        self.shape = shape
        self.ndim = len(shape)

        # dimension names
        if rec_name is None:
            self.dims_rec = []
        else:
            self.dims_rec = [rec_name]

        self.n_name = n_name
        self.dims_n = ["{}_{}".format(n_name, i) for i in range(self.ndim)]
        self.dims_lnz = ["{}_{}".format(lnz_name, i) for i in range(self.ndim)]
        self.dims_comp = [comp_name]

    @gcached()
    def coords_n(self):
        return {
            k: xr.DataArray(
                _get_range(n), dims=k, attrs={"long_name": r"${}$".format(k)}
            )
            for k, n in zip(self.dims_n, self.shape)
        }

    @gcached(prop=False)
    def attrs(self, *args):
        d = {
            "dims_n": self.dims_n,
            "dims_lnz": self.dims_lnz,
            "dims_comp": self.dims_comp,
            "dims_state": self.dims_lnz + list(args),
        }

        if self.dims_rec is not None:
            d["dims_rec"] = self.dims_rec
        return d

    @gcached(prop=False)
    def ncoords(self, coords_n=False):
        if coords_n:
            coords = self.coords_n
        else:
            coords = None
        return xr.DataArray(
            _get_indices(self.shape),
            dims=self.dims_comp + self.dims_n,
            coords=coords,
            name=self.n_name,
        )

    @gcached(prop=False)
    def ncoords_tot(self, coords_n=False):
        return self.ncoords(coords_n).sum(self.dims_comp)

    def wrap_data(self, data, coords_n=False, coords=None, **kwargs):
        if coords is None:
            coords = {}

        if coords_n:
            coords = dict(coords, **self.coords_n)

        return xr.DataArray(
            data,
            dims=self.dims_rec + self.dims_n,
            coords=coords,
            attrs=self.attrs(*kwargs.keys()),
        )

    def wrap_lnpi(self, lnpi, coords_n=False, **kwargs):
        return self.wrap_data(lnpi, coords_n=coords_n, **kwargs)

    def wrap_lnz(self, lnz, **kwargs):
        return xr.DataArray(lnz, dims=self.dims_rec + self.dims_comp, **kwargs)

    def wrap_lnpi_0(self, lnpi_0, **kwargs):
        return xr.DataArray(lnpi_0, dims=self.dims_rec, **kwargs)


################################################################################
# xlnPi
# register xge property to collection classes
def xr_name(long_name=None, name=None, unstack=True, **kws):
    """
    decorator to add name, longname to xarray output
    """

    def decorator(func):
        if name is None:
            _name = func.__name__.lstrip("_")

        else:
            _name = name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs).rename(_name)
            if hasattr(self, "_standard_attrs"):
                attrs = self._standard_attrs
            else:
                attrs = {}
            attrs = dict(attrs, **kws)

            if long_name is not None:
                attrs["long_name"] = long_name
            out = out.assign_attrs(**attrs)
            if unstack and self._xarray_unstack:
                out = out.unstack()

            return out

        return wrapper

    return decorator


@lnPiMasked.decorate_accessor("xge")
@MaskedlnPiDelayed.decorate_accessor("xge")
@MaskedlnPiLegacy.decorate_accessor("xge")
@lnPiCollection.decorate_accessor("xge")
def xge_accessor(parent):
    """
    Accessor to :class:`~lnpy.xGrandCanonical`.
    """
    return xGrandCanonical(parent)


class xGrandCanonical:
    """
    :class:`DataArray` accessor to Grand Canonical properties from lnPi


    This class is primarily interacted with through the attributes ``xge`` attached
    to :class:`~lnpy.lnPiMasked` and :class:`~lnpy.lnPiCollection`.
    """

    def __init__(self, parent):
        self._parent = parent

        self._rec_name = getattr(self._parent, "_concat_dim", None)
        if self._rec_name is None:
            shape = self._parent.shape
        else:
            shape = self._parent._series.iloc[0].shape

        self._wrapper = get_xrlnPiWrapper(shape=shape, rec_name=self._rec_name)

    @property
    def _use_cache(self):
        return getattr(self._parent, "_use_cache", False)

    @property
    def _xarray_unstack(self):
        return getattr(self._parent, "_xarray_unstack", True)

    @gcached()
    def _standard_attrs(self):
        return self._wrapper.attrs(*self._parent.state_kws.keys())

    # @gcached() to much memory
    @property
    def _rec_coords(self):
        if self._rec_name is None:
            return dict(self._parent._index_dict(), **self._parent.state_kws)
        else:
            return {
                self._parent._concat_dim: self._parent.index,
                **self._parent.state_kws,
            }

    @property
    def _xarray_dot_kws(self):
        return getattr(self._parent, "_xarray_dot_kws", {})

    @property
    def ncoords(self):
        """Particle number coordinates."""
        return self._wrapper.ncoords()

    @property
    def ncoords_tot(self):
        """Total number of particles coordinates"""
        return self._wrapper.ncoords_tot()

    @property
    def dims_n(self):
        """Dimension(s) corresponding to  number of particles"""
        return self._wrapper.dims_n

    @property
    def dims_lnz(self):
        r"""Dimension corresponding to values of :math:`\ln z`"""
        return self._wrapper.dims_lnz

    @property
    def dims_comp(self):
        """Dimension of component number"""
        return self._wrapper.dims_comp

    @property
    def dims_state(self):
        """Dimensions corresponding to 'state variables'"""
        return self.pi_norm.attrs["dims_state"]

    @property
    def dims_rec(self):
        """Dimensions for replicates"""
        return self._wrapper.dims_rec

    @property
    def beta(self):
        r"""Inverse temperature :math:`\beta = 1 / (k_{\rm B} T)`"""
        return self._parent.state_kws["beta"]

    @property
    def volume(self):
        """System volume :math:`V`."""
        return self._parent.state_kws["volume"]

    @gcached()
    def coords_state(self):
        return {k: self.pi_norm.coords[k] for k in self.dims_state}

    @gcached()
    @xr_name(r"$\beta {\bf \mu}$")
    def betamu(self):
        r"""Scaled checmical potential :math:`\beta \mu`"""
        return self._wrapper.wrap_lnz(self._parent._lnz_tot, coords=self._rec_coords)

    @property
    @xr_name(r"$\ln\beta{\bf\mu}$")
    def lnz(self):
        r"""Log of activity :math:`\ln z``"""
        return self.betamu

    @xr_name(r"$\ln \Pi(n,\mu,V,T)$", unstack=False)
    def lnpi(self, fill_value=None):
        r"""DataArray view of :math:`\ln \Pi(N)`

        Notes
        -----
        This value is always in 'stacked' form.
        You must manually unstack it.
        """
        return self._wrapper.wrap_lnpi(
            self._parent._lnpi_tot(fill_value),
            coords=self._rec_coords,
            **self._parent.state_kws,
        )  # .assign_coords(**self._rec_coords)

    @gcached()
    def _pi_params(self):
        pi_norm, pi_sum, lnpi_zero = self._parent._pi_params(-np.inf)

        pi_sum = self._wrapper.wrap_lnpi_0(
            pi_sum, name="pi_sum", coords=self._rec_coords
        )

        lnpi_zero = self._wrapper.wrap_lnpi_0(
            lnpi_zero, name="lnpi_zero", coords=self._rec_coords
        )

        pi_norm = self._wrapper.wrap_lnpi(
            pi_norm, coords=self._rec_coords, **self._parent.state_kws
        )

        return pi_norm, pi_sum, lnpi_zero

    @property
    def pi_norm(self):
        r"""Normalized value of :math:`\Pi_{\rm norm}(N) = \Pi(N) / \sum_N \Pi(N)`"""
        return self._pi_params[0]

    @property
    def pi_sum(self):
        r"""Sum of unnormalized :math:`\Pi(N)`"""
        return self._pi_params[1]

    @property
    def _lnpi_zero(self):
        return self._pi_params[2]

    @property
    def lnpi_norm(self):
        r""":math:`\ln \Pi_{\rm norm}(N)`."""
        pi = self.pi_norm
        return np.log(xr.where(pi > 0, pi, np.nan))

    def _get_prop_from_extra_kws(self, prop):
        if isinstance(prop, str):
            p = self._parent
            if hasattr(self._parent, "_series"):
                p = p.iloc[0]

            if prop not in p.extra_kws:
                raise ValueError(f"{prop} not found in extra_kws")
            else:
                prop = p.extra_kws.get(prop, None)

        return prop

    def _array_or_callable_to_xarray(self, x, allow_extra_kws=True, *args, **kwargs):

        if callable(x):
            x = x(self, *args, **kwargs)
        elif allow_extra_kws:
            x = self._get_prop_from_extra_kws(x)

        if not isinstance(x, xr.DataArray):
            x = xr.DataArray(x, dims=self.dims_n)

        return x

    def _mean_pi(self, x, *args, **kwargs):

        return xr.dot(self.pi_norm, x, dims=self.dims_n, **self._xarray_dot_kws)

    @xr_name()
    def mean_pi(self, x, *args, **kwargs):
        r"""
        Calculates :math:`\overline{x} = \sum_N \Pi_{\rm norm}(N) x(N)`

        Parameters
        ----------
        x : array, callable, or str
            If callable, should have form ``x = x(self, *args, **kwargs)``.
            If string, then set `x = self.parent.extra_kws[x]`.
            Otherwise, should be an array of same shape as single lnPi.
        *args, **kwargs
            Extra arguments to `x` if passing callable
        Returns
        -------
        ave : DataArray
        x can be an array or a callable

        f(self, *args, **kwargs)

        if x is not an xr.DataArray, try to convert it to one.
        """
        x = self._array_or_callable_to_xarray(x, *args, **kwargs)
        return self._mean_pi(x, *args, **kwargs)

    def _central_moment_bar(
        self,
        x,
        y=None,
        xmom=2,
        ymom=None,
        xmom_dim="xmom",
        ymom_dim="ymom",
        *args,
        **kwargs,
    ):
        """
        calculate central moments of the form sum(Pi * (\bar{x} - <\bar{x}>) ** n)

        Note that if you pass in the

        """

        def _get_dx(xx, mom, mom_dim):
            xx = self._array_or_callable_to_xarray(xx, *args, **kwargs)
            if isinstance(mom, int):
                mom = [mom]

            mom = xr.DataArray(mom, dims=mom_dim, coords={mom_dim: mom})

            return (xx - self._mean_pi(xx)) ** mom

        dx = _get_dx(x, xmom, xmom_dim)

        if y is not None:
            dy = _get_dx(y, ymom, ymom_dim)

            dx = dx * dy

        return self._mean_pi(dx)

    def var_pi(self, x, y=None, *args, **kwargs):
        r"""Calculate Grand Canonical variance from canonical properties.

        Given x(N) and y(N), calculate

        .. math::

            {\rm var}(x, y) = \overline{(x - \overline{x}) (y - \overline{y})}

        ``x`` and ``y`` can be arrays, or callables, in which case:
        ``x = x(self, *args, **kwargs)``

        See Also
        --------
        :math:`mean_pi`

        """

        if callable(x):
            x = x(self, *args, **kwargs)

        # x_mean = self.mean_pi(x)
        # if y is None:
        #     y = x
        # if y is x:
        #     y_mean = x_mean
        # else:
        #     if callable(y):
        #         y = y(self, *args, **kwargs)
        #     y_mean = self.mean_pi(y)
        # return self.mean_pi((x - x_mean) * (y - y_mean))

        # this cuts down on memory usage
        xx = x - self._mean_pi(x)
        if y is None:
            yy = xx
        else:
            if callable(y):
                y = y(self, *args, **kwargs)
            yy = y - self._mean_pi(y)

        return xr.dot(self.pi_norm, xx, yy, dims=self.dims_n, **self._xarray_dot_kws)

    def pipe(self, func, *args, **kwargs):
        """
        Apply function to `self`
        """
        return func(self, *args, **kwargs)

    @gcached()
    @xr_name(r"${\bf n}(\mu,V,T)$")
    def nvec(self):
        r"""Average number of particles of each component :math:`\overline{{\bf N}}`"""
        return self._mean_pi(self.ncoords)

    @gcached()
    @xr_name(r"$n(\mu,V,T)$")
    def ntot(self):
        r"""Average total number of particles :math:`\overline{N}`"""
        return self._mean_pi(self.ncoords_tot)

    @property
    @xr_name(r"${\bf x}(\mu,V,T)$")
    def molfrac(self):
        r"""Average molfrac for each components :math:`{\bf x} = \overline{{\bf N}} / N`"""
        return self.nvec.pipe(lambda x: x / x.sum(self.dims_comp))

    @gcached()
    @xr_name(r"$var[{\bf n}(\mu,V,T)]$")
    def nvec_var(self):
        r"""Variance in particle number :math:`{\rm var}\, \bf{N}`"""
        return self.var_pi(self.ncoords)

    @gcached()
    @xr_name(r"$var[n(\mu,V,T)]$")
    def ntot_var(self):
        r"""Variance in total number of particles :math:`{\rm var}\, N`"""
        return self.var_pi(self.ncoords_tot)

    @property
    @xr_name(r"${\bf \rho}(\mu,V,T)$")
    def dens(self):
        r"""Density of each component :math:`{\bf \rho} = \overline{{\bf N}} / V`"""
        # NOTE: keep this here because of some internal calculations
        return self.nvec.pipe(lambda x: x / x["volume"])

    @property
    @xr_name(r"$\rho(\mu,V,T)$", standard_name="total_density")
    def dens_tot(self):
        r"""Total density :math:`\overline{N} / V`"""
        return self.ntot.pipe(lambda x: x / x["volume"])

    def max(self, *args, **kwargs):
        r""":math:`\max_N \ln \Pi(N)`"""
        return self.lnpi().max(self.dims_n)

    # def argmax(self, *args, **kwargs):
    #     return np.array(
    #         [x.local_argmax(*args, **kwargs) for x in self._parent])

    # @xr_name('distance from upper edge')
    # def edge_distance(self, ref, *args, **kwargs):
    #     """distance from endpoint"""

    #     return xr.DataArray([x.edge_distance(ref) for x in self._parent],
    #                         dims=self.dims_rec,
    #                         coords=self._rec_coords)

    @gcached(prop=False)
    def _argmax_indexer(self):
        x = self.pi_norm.values
        xx = x.reshape(x.shape[0], -1)
        idx_flat = xx.argmax(-1)
        indexer = np.unravel_index(idx_flat, x.shape[1:])
        return indexer

    @gcached()
    def _argmax_indexer_dict(self):
        return {
            k: xr.DataArray(v, dims=self._parent._concat_dim)
            for k, v in zip(self.dims_n, self._argmax_indexer())
        }

    @gcached()
    def _sample_indexer_dict(self):
        return {
            self._parent._concat_dim: xr.DataArray(
                range(len(self._parent)), dims=self._parent._concat_dim
            )
        }

    @property
    def _sample_argmax_indexer_dict(self):
        return dict(self._argmax_indexer_dict, **self._sample_indexer_dict)

    def lnpi_max(self, fill_value=None, add_n_coords=True):
        r"""maximum value of :math:`\max_N \ln \Pi(N, ...)`"""
        out = self.lnpi(fill_value).isel(**self._sample_argmax_indexer_dict)

        # NOTE : This assumes each n value corresponds to index
        # alternatively, could put through filter like
        # coords = {k : out[k].isel(**{k : v}) for k, v in self._argmax_index_dict.items()}

        if add_n_coords:
            out = out.assign_coords(**self._argmax_indexer_dict)

        return out

    def pi_norm_max(self, add_n_coords=True):
        r"""Maximum value :math:`\max_N \Pi_{\rm norm}(N, meta)`"""
        out = self.pi_norm.isel(**self._sample_argmax_indexer_dict)
        if add_n_coords:
            out = out.assign_coords(**self._argmax_indexer_dict)
        return out

    @gcached(prop=False)
    def argmax(self):
        return np.array(self._argmax_indexer()).T

    @xr_name("distance from upper edge")
    def edge_distance(self, ref, *args, **kwargs):
        """Distance from argmax(lnPi) to endpoint"""
        out = ref.edge_distance_matrix[self._argmax_indexer()]
        return xr.DataArray(out, dims=self.dims_rec, coords=self._rec_coords)

    @xr_name("distance from edge of cut value")
    def edge_distance_val(self, ref, val=None, max_frac=None, *args, **kwargs):
        """
        Calulate min distance from where self.pi_norm > val to edge

        Parameters
        ----------
        ref : lnPiMasked
            reference object to consider.
        val : float

        max_frac : bool, optional
        if not None, val = max_frac * self.pi_norm.max(self.dims_n)
        """

        if max_frac is not None:
            assert max_frac < 1.0
            val = self.pi_norm_max(add_n_coords=False) * max_frac
        else:
            assert val is not None

        e = xr.DataArray(ref.edge_distance_matrix, dims=self.dims_n)
        mask = self.pi_norm > val
        return e.where(mask).min(self.dims_n)

    @gcached(prop=False)
    @xr_name(r"$\beta \Omega(\mu,V,T)$", standard_name="grand_potential")
    def _betaOmega(self, lnpi_zero=None):
        if lnpi_zero is None:
            lnpi_zero = self._lnpi_zero
        return lnpi_zero - np.log(self.pi_sum)

    def betaOmega(self, lnpi_zero=None):
        r"""Scaled grand potential :math:`\beta \Omega`.

        Parameters
        ----------
        lnpi_zero : float or None
            if None, lnpi_zero = self.data.ravel()[0]
        """
        # Note.  Put calculation in _betaOmega
        # because so many other things
        # call it with lnpi_zero set to None
        # so end up with copies of the same
        # thing in cache
        return self._betaOmega(lnpi_zero)

    @gcached()
    @xr_name(r"${\rm PE}(\mu,V,T)$", standard_name="potential_energy")
    def PE(self):
        r"""Potential energy :math:`\overline{PE}`"""

        # if betaPE available, use that:

        if hasattr(self._parent, "_series"):
            PE = self._parent.iloc[0].extra_kws.get("PE", None)
        else:
            PE = self._parent.extra_kws.get("PE", None)

        if PE is None:
            raise AttributeError('must set "PE" in "extra_kws" of lnPiMasked')
        else:
            PE = xr.DataArray(PE, dims=self.dims_n)
        return self._mean_pi(PE)

    @xr_name(r"$\beta F(\mu,V,T)$", standard_name="helmholtz_free_energy")
    def betaF_alt(self, betaF_can, correction=True):
        r"""Alternate scaled Helmholtz free energy :math:`\beta \overline{F}`.

        Calculated using

        .. math::

            \beta \overline{F} = \sum_N [\Pi(N) \beta F(N) + C(N)]

        Parameters
        ----------
        betaF_can : array-like
            Value of :math:`F(N)`
        correction : bool, default=True
            If True, :math:`C(N) = \ln \Pi(N) `.  Otherwise, :math:`C=0`.
        """
        if correction:
            betaF_can = betaF_can + self.lnpi_norm
        # return (self.pi_norm * betaF_can).sum(self.dims_n)
        return self._mean_pi(betaF_can)

    ################################################################################
    # Other properties
    @xr_name(r"$\beta\omega(\mu,V,T)$", standard_name="grand_potential_per_particle")
    def betaOmega_n(self, lnpi_zero=None):
        r"""Grand potential per particle :math:`\beta \Omega / \overline{N}`"""
        return self.betaOmega(lnpi_zero) / self.ntot

    # @gcached(prop=False)
    @xr_name(r"$\beta p(\mu,V,T)V$")
    def betapV(self, lnpi_zero=None):
        r""":math:`\beta p V = - \beta \Omega`"""
        return -self.betaOmega(lnpi_zero)

    @gcached()
    @xr_name("mask_stable", description="True where state is most stable")
    def mask_stable(self):
        """
        Masks are True where values are stable. Only works for unstacked data.
        """
        if not self._xarray_unstack:
            # raise Value'only mask with unstack')
            pv = self.betapV()
            sample = self._parent._concat_dim
            return (
                pv.unstack(sample)
                .pipe(lambda x: x.max("phase") == x)
                .stack(sample=pv.indexes[sample].names)
                .loc[pv.indexes["sample"]]
            )
        else:
            return self.betapV().pipe(lambda x: x.max("phase") == x)

    # @gcached(prop=False)
    @xr_name(r"$\beta p(\mu,V,T)/\rho$", standard_name="compressibility_factor")
    def Z(self, lnpi_zero=None):
        r"""Compressibility factor :math:`\beta p / \rho`"""
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r"$p(\mu,V,T)$")
    def pressure(self, lnpi_zero=None):
        r"""Pressure :math:`p = -\Omega / V`"""
        return self.betapV(lnpi_zero).pipe(lambda x: x / (x["beta"] * x["volume"]))

    @property
    @xr_name(r"${\rm PE}(\mu,V,T)/n$", standard_name="potential_energy_per_particle")
    def PE_n(self):
        r"""Potential energy per particle :math:`\overline{PE}/\overline{N}`"""
        return self.PE / self.ntot

    def table(
        self,
        keys=None,
        default_keys=("nvec", "betapV", "PE_n"),
        ref=None,
        mask_stable=False,
        dim_to_suffix=None,
    ):
        """
        Create :class:`xarray.Dataset` of calculated properties.

        Parameters
        ----------
        keys : sequence of str, optional
            keys of attributes or methods of `self` to include in ouput
        default_keys : sequence of str
            Default keys to consider.
        ref : lnPiMasked, optional
            If calculating `edge_distastance`, need a reference :class:`~lnpy.lnPiMasked` object.
        mask_stable : bool, default=False
            If True, remove any unstable values
        dim_to_suffix : sequence of hashables, optional
            dimensions to remove from ouput.  These are instead added as suffix to variable names

        Returns
        -------
        ds : Dataset
            Contianing all the calculated properties in a single object

        Notes
        -----
        The results can be easily convert to a :class:`pandas.DataFrame` using ``ds.to_frame()``
        """

        out = []
        if ref is not None:
            out.append(self.edge_distance(ref))

        def _process_keys(x):
            if x is None:
                return []
            elif isinstance(x, str):
                return [x]
            else:
                return list(x)

        keys = _process_keys(keys) + _process_keys(default_keys)
        if not keys:
            raise ValueError("must specify some keys or default_keys to use")

        # only unique keys
        # this preserves order
        for key in dict.fromkeys(keys):
            try:
                v = getattr(self, key)
                if callable(v):
                    v = v()
                out.append(v)
            except Exception:
                pass

        out = xr.merge(out)
        if "lnz" in keys:
            # if including property lnz, then drop lnz_0, lnz_1,...
            out = out.drop(out["lnz"]["dims_lnz"])

        if mask_stable:
            # mask_stable inserts nan in non-stable
            mask_stable = self.mask_stable
            if not self._xarray_unstack:
                out = out.where(mask_stable, drop=True)
            else:
                phase = out.phase
                out = (
                    out.where(mask_stable)
                    .max("phase")
                    .assign_coords(phase=lambda x: phase[mask_stable.argmax("phase")])
                )

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                out = out.pipe(dim_to_suffix_dataset, dim=dim)
        return out

    @property
    @xr_name(r"$\beta G(\mu,V,T)$", standard_name="Gibbs_free_energy")
    def betaG(self):
        r"""Scaled Gibbs free energy :math:`\beta G = \sum_i \beta \mu_i \overline{N}_i`."""
        # return self.betamu.pipe(lambda betamu: (betamu * self.nvec).sum(betamu.dims_comp))
        # return (self.betamu * self.nvec).sum(self.dims_comp)
        return xr.dot(
            self.betamu, self.nvec, dims=self.dims_comp, **self._xarray_dot_kws
        )

    @property
    @xr_name(r"$\beta G(\mu,V,T)/n$", standard_name="Gibbs_free_energy_per_particle")
    def betaG_n(self):
        r"""Scaled Gibbs free energy per particle :math:`\beta G / \overline{N}`."""
        return self.betaG / self.ntot

    @xr_name(r"$\beta F(\mu,V,T)$", standard_name="helmholtz_free_energy")
    def betaF(self, lnpi_zero=None):
        r"""Scaled Helmholtz free energy :math:`\beta F = \beta \Omega + \beta G`."""
        return self.betaOmega(lnpi_zero) + self.betaG

    @xr_name(
        r"$\beta F(\mu,V,T)/n$", standard_name="helmholtz_free_energy_per_particle"
    )
    def betaF_n(self, lnpi_zero=None):
        r"""Scaled Helmholtz free enenergy per particle :math:`\beta F / \overline{N}`."""
        return self.betaF(lnpi_zero) / self.ntot

    @xr_name(r"$\beta E(\mu,V,T)$", standard_name="total_energy")
    def betaE(self, ndim=3):
        r"""Scaled total energy :math:`\beta E = \beta \overline{PE} + \beta \overline{KE}`."""
        return ndim / 2.0 * self.ntot + self.PE.pipe(lambda x: x * x.beta)

    @xr_name(r"$\beta E(\mu,V,T)/ n$", standard_name="total_energy_per_particle")
    def betaE_n(self, ndim=3):
        r"""Scaled total energy per particle :math:`\beta E / \overline{N}`."""
        return self.betaE(ndim) / self.ntot

    @xr_name(r"$S(\mu,V,T) / k_{\rm B}$", standard_name="entropy")
    def S(self, lnpi_zero=None, ndim=3):
        r"""Scaled entropy :math:`S / k_{\rm B} = \beta E - \beta F`."""
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r"$S(\mu,V,T)/(n kB)$", standard_name="entropy_per_particle")
    def S_n(self, lnpi_zero=None, ndim=3):
        r"""Scaled entropy per particle :math:`S / (N k_{\rm B})`."""

        return self.S(lnpi_zero, ndim) / self.ntot


# NOTE: Using function accessor to override the docstring
@lnPiMasked.decorate_accessor("xce")
@MaskedlnPiDelayed.decorate_accessor("xce")
@MaskedlnPiLegacy.decorate_accessor("xce")
def xce_accessor(parent):
    """Accessor to :class:`~lnpy.xCanonical`"""
    return xCanonical(parent)


class xCanonical(object):
    """
    Canonical ensemble properties

    Parameters
    ----------
    parent : lnPiMasked
    """

    def __init__(self, parent):
        self._parent = parent
        self._xge = parent.xge

    def lnpi(self, fill_value=None):
        r"""DataArray view of :math:`\ln Pi(N)`"""
        return self._xge.lnpi(fill_value=fill_value)

    @property
    def _xarray_unstack(self):
        return getattr(self._parent, "_xarray_unstack", True)

    @gcached()
    def ncoords(self):
        """Coordinate vector `dims_n`"""
        return (
            self._xge.ncoords
            # NOTE: if don't use '.values', then get extra coords don't want
            .where(~self._xge.lnpi(np.nan).isnull().values)
        )

    @property
    def nvec(self):
        r"""Number of particles for each components :math:`{\bf N}`"""
        return self.ncoords.rename("nvec")

    @gcached()
    def ntot(self):
        """Total number of particles :math:`N`"""
        return self.ncoords.sum(self._xge.dims_comp)

    @gcached(prop=False)
    @xr_name(r"$\beta F({\bf n},V,T)$", standard_name="helmholtz_free_energy")
    def _betaF(self, lnpi_zero=None):
        """
        Helmholtz free energy
        """
        x = self._parent.xge

        if lnpi_zero is None:
            lnpi_zero = self._parent.data.ravel()[0]
            # lnpi_zero = x.lnpi_zero

        return (
            (-(x.lnpi(np.nan) - lnpi_zero) + (x.ncoords * x.betamu).sum(x.dims_comp))
            .assign_coords(**x._wrapper.coords_n)
            .drop(x.dims_lnz)
            .assign_attrs(x._standard_attrs)
        )

    def betaF(self, lnpi_zero=None):
        r"""Scaled Helmoltz free energy :math:`\beta F`"""
        return self._betaF(lnpi_zero)

    @xr_name(
        r"$\beta F({\bf n},V,T)/n$", standard_name="helmholtz_free_energy_per_particle"
    )
    def betaF_n(self, lnpi_zero=None):
        r"""Scaled Helmoltz free energy per particle :math:`\beta F / N`"""
        return self.betaF(lnpi_zero) / self.ntot

    @gcached()
    @xr_name(r"${\rm PE}({\bf n},V,T)/n$", standard_name="potential_energy")
    def PE(self):
        """Internal Energy :math:`PE`"""
        # if betaPE available, use that:
        PE = self._parent.extra_kws.get("PE", None)
        if PE is None:
            raise AttributeError('must set "PE" in "extra_kws" of lnPiMasked')
        x = self._xge
        coords = dict(x._wrapper.coords_n, **self._parent.state_kws)
        return xr.DataArray(PE, dims=x.dims_n, coords=coords, attrs=x._standard_attrs)

    @property
    @xr_name(
        r"${\rm PE}({\bf n},V,T)/n$", standard_name="potential_energy_per_particle"
    )
    def PE_n(self):
        r"""Internal energy per particle :math:`PE / N`"""
        return self.PE / self.ntot

    @xr_name(r"$\beta E({\bf n},V,T)$")
    def betaE(self, ndim=3):
        r"""Scaled total energy :math:`\beta E = \beta PE + \beta KE`"""
        return ndim / 2 * self.ntot + self._xge.beta * self.PE

    @xr_name(r"$\beta E({\bf n},V,T)/n$")
    def betaE_n(self, ndim=3):
        """Scaled total energy per particle"""
        return self.betaE(ndim) / self.ntot

    @xr_name(r"$S({\bf n},V,T)/k_{\rm B}$")
    def S(self, ndim=3, lnpi_zero=None):
        r"""Entropy :math:`S / k_{\rm B}`"""
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r"$S({\bf n},V,T)/(n k_{rm B})$")
    def S_n(self, ndim=3, lnpi_zero=None):
        r"""Entropy per particle :math:`S / (N k_{\rm B})`"""
        return self.S(ndim, lnpi_zero) / self.ntot

    @gcached(prop=False)
    @xr_name(r"$\beta {\bf\mu}({bf n},V,T)$", standard_name="absolute_activity")
    def _betamu(self, lnpi_zero=None):
        return xr.concat(
            [self.betaF(lnpi_zero).differentiate(n) for n in self._xge.dims_n],
            dim=self._xge.dims_comp[0],
        ).assign_attrs(self._xge._standard_attrs)

    def betamu(self, lnpi_zero=None):
        r"""Scaled chemical potential :math:`\beta \mu`"""
        return self._betamu(lnpi_zero)

    @property
    @xr_name(r"${\bf \rho}({\bf n},V,T)$")
    def dens(self):
        r"""Density :math:`\rho = N / V`"""
        return self.ncoords / self._xge.volume

    @gcached(prop=False)
    @xr_name(r"$\beta\Omega({\bf n},V,T)$")
    def _betaOmega(self, lnpi_zero=None):
        """
        beta * Omega = betaF - lnz .dot. N
        """
        return self.betaF(lnpi_zero) - (self.betamu(lnpi_zero) * self.ncoords).sum(
            self._xge.dims_comp
        )

    def betaOmega(self, lnpi_zero=None):
        r"""Scaled Grand Potential :math:`\beta \Omega`"""
        return self._betaOmega(lnpi_zero)

    @xr_name(r"$\beta\Omega({\bf n},V,T)/n$")
    def betaOmega_n(self, lnpi_zero=None):
        r"""Scaled Grand potential per particle, :math:`\beta\Omega / N`"""
        return self.betaOmega(lnpi_zero) / self.ntot

    @xr_name(r"$\beta p({\bf n},V,T)V$")
    def betapV(self, lnpi_zero=None):
        r""":math:`\beta p V = -\beta \Omega`"""
        return -self.betaOmega(lnpi_zero)

    @xr_name(r"$\beta p({\bf n},V,T)/\rho$")
    def Z(self, lnpi_zero=None):
        r"""Compressibility :math:`Z = \beta p / \rho = pV / (k_{\rm B} T)`"""
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r"$p({\bf n},V,T)$")
    def pressure(self, lnpi_zero=None):
        r"""Pressure :math:`p = -\Omega / V`"""
        return self.betapV(lnpi_zero).pipe(lambda x: x / (x.beta * x.volume))

    def table(
        self,
        keys=None,
        default_keys=("betamu", "betapV", "PE_n", "betaF_n"),
        dim_to_suffix=None,
    ):
        """
        Create Dataset from attributes/methods of `self`

        Parameters
        ----------
        keys : sequence of str
            Names of attributes/methods to access
        default_keys : sequence of str
            Default keys to access.
        dim_to_suffix : str, sequence of str, optional
            If passed, convert dimenensions in `dim_to_suffix` from dimensions in output to suffixes to variable names


        Returns
        -------
        table : Dataset
        """

        out = []

        def _process_keys(x):
            if x is None:
                return []
            elif isinstance(x, str):
                return [x]
            else:
                return list(x)

        keys = _process_keys(keys) + _process_keys(default_keys)

        # loop over unique keys only, in order
        for key in dict.fromkeys(keys):
            try:
                v = getattr(self, key)
                if callable(v):
                    v = v()
                out.append(v)
            except Exception:
                pass

        out = xr.merge(out)

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                out = out.pipe(dim_to_suffix_dataset, dim=dim)
        return out
