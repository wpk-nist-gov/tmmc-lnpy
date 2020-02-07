from __future__ import absolute_import

import numpy as np
import pandas as pd
from scipy.ndimage import filters

from .cached_decorators import gcached
from .utils import labels_to_masks, masks_change_convention

from .extensions import AccessorMixin


################################################################################
# Delayed
from functools import lru_cache

@lru_cache(maxsize=20)
def _get_n_ranges(shape, dtype):
    return [np.arange(s, dtype=dtype) for s in shape]

@lru_cache(maxsize=20)
def _get_shift(shape, dlnz, dtype):
    shift = np.zeros([], dtype=dtype)
    for i, (nr, m) in enumerate(zip(_get_n_ranges(shape, dtype), dlnz)):
        shift = np.add.outer(shift, nr * m)
    return shift

@lru_cache(maxsize=20)
def _get_data(self, dlnz):
    if all((x==0 for x in dlnz)):
        return self._data
    else:
        return _get_shift(self.shape, dlnz, self._data.dtype) + self._data

@lru_cache(maxsize=20)
def _get_maskedarray(self, dlnz):
    return np.ma.MaskedArray(_get_data(self, dlnz), mask=self._mask, fill_value=self._fill_value)

@lru_cache(maxsize=20)
def _get_filled(self, dlnz, fill_value=None):
    return _get_maskedarray(self, dlnz).filled(fill_value)


class _MaskedlnPiDelayedBase(object):
    def __init__(self, lnz, data, mask=None,
                 state_kws=None, extra_kws=None, fill_value=np.nan,
                 copy_data=False, copy_mask=False):

        lnz = np.atleast_1d(lnz)
        data = np.array(data, copy=copy_data)
        assert data.ndim == len(lnz)
        if mask is None:
            mask = np.full(data.shape, fill_value=False, dtype=np.bool)
        else:
            mask = np.array(mask, copy=copy_mask, dtype=np.bool)
        assert mask.shape == data.shape

        if state_kws is None:
            state_kws = {}
        if extra_kws is None:
            extra_kws = {}

        self._data = data
        self._mask = mask

        self._state_kws = state_kws
        self._extra_kws = extra_kws

        self._lnz = lnz
        self._lnz_data = lnz_data
        self._fill_value = fill_value
        self._dlnz = tuple(self._lnz - self._lnz_data)







class MaskedlnPiDelayed(AccessorMixin):

    def __init__(self, lnz, lnz_data, data, mask=None,
                 state_kws=None, extra_kws=None, fill_value=np.nan,
                 copy_data=False, copy_mask=False):

        lnz = np.atleast_1d(lnz)
        lnz_data = np.atleast_1d(lnz_data)
        assert lnz.shape == lnz_data.shape

        data = np.array(data, copy=copy_data)
        assert data.ndim == len(lnz)

        if mask is None:
            mask = np.full(data.shape, fill_value=False, dtype=np.bool)
        else:
            mask = np.array(mask, copy=copy_mask, dtype=np.bool)
        assert mask.shape == data.shape


        if state_kws is None:
            state_kws = {}
        if extra_kws is None:
            extra_kws = {}

        self._data = data
        self._mask = mask

        self._state_kws = state_kws
        self._extra_kws = extra_kws

        self._lnz = lnz
        self._lnz_data = lnz_data
        self._fill_value = fill_value
        self._dlnz = tuple(self._lnz - self._lnz_data)

    @property
    def dtype(self):
        return self._data.dtype

    def _clear_cache(self):
        self._cache = {}

    @property
    def state_kws(self):
        return self._state_kws

    @property
    def extra_kws(self):
        return self._extra_kws

    @property
    def ma(self):
        return _get_maskedarray(self, self._dlnz)

    def filled(self, fill_value=None):
        return _get_filled(self, self._dlnz, fill_value)

    @property
    def data(self):
        return _get_data(self, self._dlnz)

    @property
    def mask(self):
        return self._mask

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return len(self._data)

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def lnz(self):
        return self._lnz

    @property
    def betamu(self):
        return self._lnz

    @property
    def volume(self):
        return self.state_kws.get('volume', None)

    @property
    def beta(self):
        return self.state_kws.get('beta', None)

    def __repr__(self):
        return '<lnPi(lnz={})>'.format(self._lnz)

    def __str__(self):
        repr(self)

    def _index_dict(self, phase=None):
        out = {'lnz_{}'.format(i): v for i, v in enumerate(self.lnz)}
        if phase is not None:
            out['phase'] = phase
        #out.update(**self.state_kws)
        return out


    # Parameters for xlnPi
    def _lnpi_tot(self, fill_value=None):
        return self.filled(fill_value)

    def _pi_params(self, fill_value=None):
        lnpi = self._lnpi_tot(fill_value)

        lnpi_local_max = lnpi.max()
        pi = np.exp(lnpi - lnpi_local_max)
        pi_sum = pi.sum()
        pi_norm = pi / pi_sum

        lnpi_zero = self.data.ravel()[0] - lnpi_local_max

        return pi_norm, pi_sum, lnpi_zero

    @property
    def _lnz_tot(self):
        return self.lnz

    @gcached(prop=False)
    def local_argmax(self, *args, **kwargs):
        return np.unravel_index(self.ma.argmax(*args, **kwargs), self.shape)

    @gcached(prop=False)
    def local_max(self, *args, **kwargs):
        return self.ma[self.local_argmax(*args, **kwargs)]

    @gcached(prop=False)
    def local_maxmask(self, *args, **kwargs):
        return self.ma == self.local_max(*args, **kwargs)

    @gcached()
    def edge_distance_matrix(self):
        """matrix of distance from upper bound"""
        from .utils import distance_matrix
        return distance_matrix(~self.mask)

    def edge_distance(self, ref, *args, **kwargs):
        return ref.edge_distance_matrix[self.local_argmax(*args, **kwargs)]


    def new_like(self, lnz=None, data=None, mask=None,
                 copy_data=False, copy_mask=False):

        if lnz is None:
            lnz = self._lnz
        if data is None:
            data = self._data
        if mask is None:
            mask = self._mask

        return self.__class__(lnz=lnz,
                              lnz_data=self._lnz_data,
                              data=data,
                              mask=mask,
                              state_kws=self._state_kws,
                              extra_kws=self._extra_kws,
                              fill_value=self._fill_value,
                              copy_data=copy_data,
                              copy_mask=copy_mask)

    def pad(self,
            axes=None,
            ffill=True,
            bfill=False,
            limit=None):
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
        from .utils import ffill, bfill
        import bottleneck

        if axes is None:
            axes = range(self.ndim)

        data = self._data
        datas = []

        if ffill:
            datas += [ffill(data, axis=axis, limit=limit) for axis in axes]
        if bfill:
            datas += [bfill(data, axis=axis, limit=limit) for axis in axes]

        if len(datas) > 0:
            data = bottleneck.nanmean(datas, axis=0)

        new = self.new_like(data=data)
        return new

    def zeromax(self):
        """
        shift so that lnpi.max() == 0 on reference
        """

        data = self._data - np.ma.MaskedArray(self._data, mask=self._mask).max()
        return self.new_like(data=data)


    def reweight(self, lnz):
        return self.new_like(lnz=lnz)


    def or_mask(self, mask, **kwargs):
        """
        new object with logical or of self.mask and mask
        """
        return self.new_like(mask=(mask|self.mask))

    def and_mask(self, mask, **kwargs):
        """
        new object with logical and of self.mask and mask
        """
        return self.copy_shallow(mask=(mask & self.mask))

    @classmethod
    def from_table(cls,
                   path,
                   lnz,
                   state_kws=None,
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
        lnz : array-like
            beta*(chemical potential) for each component
        state_kws : dict, optional
            define state variables, like volume, beta
        sep : string, optional
            separator for file read
        names : column names
        csv_kws : dict, optional
            optional arguments to `pandas.read_csv`
        kwargs  : extra arguments
            Passed to lnPi constructor
        """
        lnz = np.atleast_1d(lnz)
        ndim = len(lnz)

        if names is None:
            names = ['n_{}'.format(i) for i in range(ndim)] + ['lnpi']

        if csv_kws is None:
            csv_kws = {}

        da = (pd.read_csv(path, sep=sep, names=names,
                          **csv_kws).set_index(names[:-1])['lnpi'].to_xarray())
        return cls(data=da.values,
                   mask=da.isnull().values,
                   lnz=lnz,
                   lnz_data=lnz,
                   state_kws=state_kws,
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
            state_as_attrs = bool(da.attrs.get('state_as_attrs', False))
        if state_as_attrs:
            # state variables from attrs
            c = da.attrs
        else:
            c = da.coords

        lnz = []
        state_kws = {}
        for k in da.attrs['dims_state']:
            val = np.array(c[k])
            if 'lnz' in k:
                lnz.append(val)
            else:
                state_kws[k] = val * 1
        kws['lnz'] = lnz
        kws['state_kws'] = state_kws

        kws['lnz_data'] = lnz

        # any overrides
        kwargs = dict(kws, **kwargs)
        return cls(**kwargs)

    def list_from_masks(self, masks, convention='image'):
        """
        create list of lnpis corresponding to masks[i]

        Parameters
        ----------
        masks : list
            masks[i] is the mask for i'th lnpi
        convention : str or bool
            convention of input masks
        Returns
        -------
        lnpis : list
            list of lnpis corresponding to each mask
        """

        return [
            self.or_mask(m)
            for m in masks_change_convention(masks, convention, False)
        ]

    def list_from_labels(self,
                         labels,
                         features=None,
                         include_boundary=False,
                         check_features=True,
                         **kwargs):
        """
        create list of lnpis corresponding to labels
        """

        masks, features = labels_to_masks(labels=labels,
                                          features=features,
                                          include_boundary=include_boundary,
                                          convention=False,
                                          check_features=check_features,
                                          **kwargs)
        return self.list_from_masks(masks, convention=False)






