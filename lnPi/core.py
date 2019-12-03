from __future__ import print_function, absolute_import, division

from collections import Iterable
from functools import partial

import numpy as np
import xarray as xr
import pandas as pd

from scipy.ndimage import filters

from .cached_decorators import gcached, cached_clear
from .utils import labels_to_masks, masks_to_labels, masks_change_convention

from .extensions import AccessorMixin, ListAccessorMixin
from .extensions import decorate_listproperty, decorate_listaccessor
# NOTE : This is a rework of core.
# [ ] : split xarray functionality into wrapper(s)
# [ ] : split splitting into separate classes


class MaskedlnPi(np.ma.MaskedArray, AccessorMixin):
    """
    class to store masked ln[Pi(n0,n1,...)].
    shape is (N0,N1,...) where Ni is the span of each dimension)

    Attributes
    ----------
    self : masked array containing lnPi
    lnz : log(absolute activity) = beta * (chemical potential) for each component
    coords : coordinate array (ndim,N0,N1,...)
    pi : exp(lnPi)
    grand : grand potential (-pV) of system
    argmax_local : local argmax (in np.where output form)
    zeromax : set lnPi = lnPi - lnPi.max()
    pad : fill in masked points by interpolation
    adjust : zeromax and/or pad
    reweight : create new lnPi at new mu
    smooth : create smoothed object
    """

    def __new__(cls,
                data,
                lnz=None,
                state_kws=None,
                extra_kws=None,
                **kwargs):
        """
        constructor

        Parameters
        ----------
        data : array-like
         data for lnPi

        lnz : array-like (Default None)
            if None, set lnz=np.zeros(data.ndim)
        state_kws : dict, optional
            dictionary of state values, which will be carried along down the road, such as `volume` and `beta`.
            These parameters will be pushed to `self.xgce` coordinates.
        extra_kws : dict, optional
            this defines extra parameters to pass along.
            Note that for potential energy calculations, extra_kws should contain
            `PE` (total potentail energy for each N vector)
        zeromax : bool (Default False)
            if True, shift lnPi = lnPi - lnPi.max()
        pad : bool (Default False)
            if True, pad masked region by interpolation
        kwargs : arguments to np.ma.array
            e.g., mask=...
        """

        obj = np.ma.array(data, **kwargs).view(cls)
        fv = kwargs.get('fill_value', None) or getattr(data, 'fill_value', None)
        if fv is None:
            fv = np.nan
        obj.set_fill_value(fv)

        # make sure to broadcase mask if it is just False
        if obj.mask is False:
            obj.mask = False

        # set mu value:
        if lnz is None:
            lnz = np.zeros(obj.ndim)
        lnz = np.atleast_1d(lnz).astype(obj.dtype)
        if len(lnz) != obj.ndim:
            raise ValueError('bad len on lnz %s' % lnz)



        if state_kws is None:
            state_kws = {}
        if extra_kws is None:
            extra_kws = {}

        obj._optinfo.update(
            lnz=lnz,
            state_kws=state_kws,
            extra_kws=extra_kws,
        )
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
        """all extra properties"""
        return self._optinfo
    @property
    def state_kws(self):
        """state specific parameters"""
        return self._optinfo['state_kws']
    @property
    def extra_kws(self):
        """all extra parameters"""
        return self._optinfo['extra_kws']

    @property
    def lnz(self):
        return self._optinfo.get('lnz', None)
    @property
    def betamu(self):
        return self.lnz


    # @property
    # def mu(self):
    #     return self._optinfo.get('mu', None)
    @property
    def volume(self):
        return self.state_kws.get('volume', None)
    @property
    def beta(self):
        return self.state_kws.get('beta', None)



    def __repr__(self):
        L = []
        L.append('lnz={}'.format(repr(self.lnz)))
        L.append('state_kws={}'.format(repr(self.state_kws)))

        if len(self.extra_kws) > 0:
            L.append('extra_kws={}'.format(repr(self.extra_kws)))

        L.append('data={}'.format(super(MaskedlnPi,self).__repr__()))

        indent = ' ' * 5
        p = 'MaskedlnPi(\n' + '\n'.join([indent + x for x in L]) + '\n)'

        return p

    @gcached(prop=False)
    def local_argmax(self, *args, **kwargs):
        return np.unravel_index(self.argmax(*args, **kwargs), self.shape)
    @gcached(prop=False)
    def local_max(self, *args, **kwargs):
        return self[self.local_argmax(*args, **kwargs)]
    @gcached(prop=False)
    def local_maxmask(self, *args, **kwargs):
        return self == self.local_max(*args, **kwargs)


    @gcached()
    def edge_distance_matrix(self):
        """matrix of distance from upper bound"""
        from .utils import distance_matrix
        return distance_matrix(~self.mask)
    def edge_distance(self, ref, *args, **kwargs):
        return ref.edge_distance_matrix[self.local_argmax(*args, **kwargs)]



    # make these top level
    #@gcached()
    @property
    def pi(self):
        """
        basic pi = exp(lnpi)
        """
        pi = np.exp(self - self.local_max())
        return pi

    @gcached()
    def pi_sum(self):
        return self.pi.sum()

    @gcached(prop=False)
    def betaOmega(self, zval=None):
        if zval is None:
            zval = self.data.ravel()[0] - self.local_max()
        return  (zval - np.log(self.pi_sum))

    def __setitem__(self, index, value):
        self._clear_cache()
        super().__setitem__(index, value)

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
        from .utils import ffill, bfill
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


    def reweight(self, lnz, zeromax=False, pad=False):
        """
        get lnpi at new lnz

        Parameters
        ----------
        lnz : array-like
            chem. pot. for new state point

        zeromax : bool (Default False)

        pad : bool (Default False)

        phases : dict


        Returns
        -------
        lnPi(lnz)
        """

        lnz = np.atleast_1d(lnz)

        assert(len(lnz) == len(self.lnz))

        new = self.copy()
        new._optinfo['lnz'] = lnz


        dlnz = new.lnz - self.lnz

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
        for i, (s, m) in enumerate(zip(self.shape, dlnz)):
            shift = np.add.outer(shift, np.arange(s) * m)

        #scale by beta
        #shift *= self.beta

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
        return cls(
            data=da.values,
            mask=da.isnull().values,
            lnz=lnz,
            state_kws=state_kws, **kwargs)



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

        return [self.or_mask(m) for m in
                masks_change_convention(masks, convention, False)]

    def list_from_labels(self, labels, features=None, include_boundary=False,
                         check_features=True, **kwargs):
        """
        create list of lnpis corresponding to labels
        """

        masks, features = labels_to_masks(labels=labels, features=features,
                                                  include_boundary=include_boundary,
                                                  convention=False, check_features=check_features,
                                                  **kwargs)
        return self.list_from_masks(masks, convention=False)


class BaselnPiCollection(AccessorMixin, ListAccessorMixin):
    """
    collection of phases
    """

    _CONCAT_DIM = None #'phase'
    _CONCAT_COORDS = 'different'

    def __init__(self, items, index=None, xarray_output=True, concat_dim=None, concat_coords=None):
        self.items = items
        self.index = index
        self.xarray_output = xarray_output

        # maybe reset concat dims/coords
        if concat_dim is not None:
            self._CONCAT_DIM = concat_dim
        if concat_coords is not None:
            self._CONCAT_COORDS = concat_coords


    @property
    def items(self):
        return self._items

    @items.setter
    @cached_clear()
    def items(self, items):
        if not isinstance(items, Iterable):
            items = [items]
        self._items = list(items)


    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if index is None:
            index = np.arange(len(self))
        else:
            assert len(index) == len(self)
        self._index = pd.Index(index, name=self._CONCAT_DIM)

    def __getitem__(self, idx):
        return self.items[idx]

    def __setitem__(self, idx, val):
        self._cache = {}
        self._items[idx] = val

    def __len__(self):
        return len(self.items)

    def wrap_list_results(self, items):
        if self.xarray_output:
            x = items[0]
            if isinstance(x, xr.DataArray):
                return (
                    xr.concat(items, self.index, coords=self._CONCAT_COORDS)
                )
        # else try to make array
        try:
            return np.array(items)
        except:
            return itmes


    def copy(self):
        return self.__class__(items=self.items, index=self.index, xarray_output=self.xarray_output)

    @cached_clear()
    def append(self, val, index=None):
        if not issubclass(type(val), self[0].__class__):
            raise ValueError(
                'can only append subclasses of type '.format(
                    type(self[0].__class__)))
        if index is None:
            index = self.index[-1] + 1
        if not isinstance(index, list):
            index = [index]
        self._items.append(val)
        self.index = self.index.to_list() + index

    @cached_clear()
    def extend(self, vals, index=None):
        # two cases
        # vals is same type as self
        # vals is list of same type of self[0]...

        if isinstance(vals, self.__class__):
            items = vals.items
            if index is None:
                index = vals.index.to_list()
        elif isinstance(vals, list):
            # list of stuff
            # check if all items are of right type
            for val in vals:
                assert(issubclass(type(val), self[0].__class__))
            items = vals
            if index is None:
                offset = self.index[-1]+1
                index = [x+offset for x in range(len(vals))]
        else:
            raise ValueError(
                'can only extend with type {} or list of subclasses of type {}'.format(
                    self.__class__,
                    self[0].__class__,
                ))

        assert len(index) == len(items)
        self._items.extend(items)
        self.index = self.index.to_list() + index

    def __add__(self, vals):
        # if adding something
        # save requirements as extend
        if isinstance(vals, self.__class__):
            items = vals.items
            index = self.index.to_list()
        elif isinstance(vals, list):
            # check vals
            for val in vals:
                assert(issubclass(type(val), self[0].__class__))
            items = vals
            offset = self.index[-1] + 1
            index = [x+offset for x in range(len(vals))]
        else:
            raise ValueError(
                'can add/extend self with with type {} or list of subclasses of type {}'.format(
                    self.__class__,
                    self[0].__class__,
                ))

        return self.__class__(items=self.items + items,
                              index=self.index.to_list() + index,
                              xarray_output=self.xarray_output)

    def __iadd__(self, x):
        self.extend(x)
        return self


    def __repr__(self):
        header = f"<{self.__class__.__name__}  ({self._CONCAT_DIM} : {len(self)})>"
        index = f"index {repr(self.index)}"
        return '\n'.join([header, index])




class Phases(BaselnPiCollection):
    _CONCAT_DIM = 'phase'
    _CONCAT_COORDS = 'different'

    # @property
    # def phases(self):
    #     """provide top level access to collections of phases"""
    #     return self

    @gcached()
    def _series(self):
        """
        series representation of items
        """
        return pd.Series(self.items, index=self.index)

    def _check_items(self):
        pass

    @classmethod
    def from_masks(cls, ref, masks, convention='image', index=None, **kwargs):
        """
        create Phases from masks
        """
        items = ref.list_from_masks(masks, convention=convention)
        return cls(items=items, index=index, **kwargs)


    @classmethod
    def from_labels(cls, ref, labels, lnz=None, features=None, include_boundary=False, labels_kws=None, check_features=True, **kwargs):
        """
        create PHases from labels
        """
        if labels_kws is None:
            labels_kws = {}
        if lnz is not None:
            ref = ref.reweight(lnz)
        masks, features = labels_to_masks(labels=labels, features=features, include_boundary=include_boundary, convention=False, check_features=check_features, **labels_kws)
        index = np.array(features) - 1

        items = ref.list_from_masks(masks, convention=False)

        return cls(items=items, index=index, **kwargs)

    @gcached()
    def labels(self):
        """labels corresponding to masks"""
        features = np.array(self.index) + 1
        return masks_to_labels([x.mask for x in self], features=features, convention=False, dtype=np.int8)

    def to_dataarray(self, dtype=np.uint8, **kwargs):
        """
        create dataarray object from labels
        """

        data = self.labels
        if dtype is not None:
            data = data.astype(dtype)
        return xr.DataArray(
            data,
            dims=self[0].xgce.dims_n,
            name='labels',
            coords=self[0].xgce.coords_state,
            attrs=self[0].xgce.lnpi.attrs
        )

    @classmethod
    def from_dataarray(cls, ref, da, lnz=None, include_boundary=False, labels_kws=None, **kwargs):
        labels = da.values
        ndim = labels.ndim

        if lnz is None:
            lnz = [da.coords[k] * 1.0 for k in da.attrs['dims_lnz']]

        return cls.from_labels(ref=ref, labels=labels, lnz=lnz, include_boundary=include_boundary, labels_kws=labels_kws, **kwargs)


    # properties
    @property
    def lnz(self):
        return self[0].lnz


# add in accessor to property lnz
@decorate_listproperty(['lnz'])
class CollectionPhases(BaselnPiCollection):
    _CONCAT_DIM = 'rec'
    _CONCAT_COORDS = 'all'

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(items=self.items[idx], index=self.index[idx], xarray_output=self.xarray_output)
        else:
            return self.items[idx]

    @gcached()
    def _series(self):
        return pd.concat({rec: x._series for rec, x in zip(self.index, self.items)}, names=self.index.names)



    def sort_by_lnz(self, comp=0, inplace=True, **kwargs):
        order = np.argsort(self.lnz[:, comp])
        L = [self._items[i] for i in order]

        if inplace:
            self._items = L
            self._cache = {}
        else:
            return self.__class__(items=L, **kwargs)
        

    ##################################################
    #builders
    ##################################################
    @classmethod
    def from_lnz_iter(cls, lnzs, ref=None, build_phases=None, build_phases_kws=None,  nmax=2, xarray_output=True, **kwargs):
        """
        build Collection from lnzs

        Parameters
        ----------
        ref : lnpi_phases object
            lnpi_phases to reweight to get list of lnpi's
        lnzs : iterable
            chem. pots. to get lnpi
        build_phases : callable
            function to create phases object
            if None, then use `DefaultBuilder`.
        build_phases_kws : optional
            optional arguments to `build_phases`

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : Collection object
        """
        if build_phases is None:
            raise ValueError('must supply build_phases')
        if build_phases_kws is None:
            build_phases_kws = {}
        L = [build_phases(ref=ref, lnz=lnz, **build_phases_kws) for lnz in lnzs]
        return cls(items=L, index=None, xarray_output=xarray_output)

    @classmethod
    def from_lnz(cls, lnz, x, ref=None, build_phases=None, build_phases_kws=None, nmax=2,  xarray_output=True, **kwargs):
        """
        build Collection from lnz builder

        Parameters
        --------- 
        ref : lnpi object
            lnpi to reweight to get list of lnpi's

        lnz : list
            list with one element equal to None.
            This is the component which will be varied
            For example, lnz=[lnz0,None,lnz2] implies use values
            of lnz0,lnz2 for components 0 and 2, and vary component 1

        x : array
            values to insert for variable component

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : Collection object
        """
        from .utils import get_lnz_iter

        lnzs = get_lnz_iter(lnz, x)
        return cls.from_lnz_iter(ref=ref, lnzs=lnzs,
                                build_phases=build_phases,
                                build_phases_kws=build_phases_kws,
                                nmax=nmax,
                                xarray_output=xarray_output, **kwargs)

    def to_dataarray(self,
                     dtype=np.uint8,
                     **kwargs):
        da = self.wrap_list_results(
            [x.to_dataarray(dtype=dtype, **kwargs) for x in self])




        # add in spinodal/binodal
        # for k in ['spinodals', 'binodals']:
        #     _k = '_' + k
        #     label = np.zeros(len(self), dtype=dtype)
        #     if hasattr(self, _k):
        #         for i, target in enumerate(getattr(self, _k)):
        #             if target is None:
        #                 break
        #             for rec, x in enumerate(self):
        #                 if x is target:
        #                     label[rec] = i + 1
        #     # else no mark
        #     da.coords[k] = (dim, label)
        return da


    # binodal/spinodal stuff
    @classmethod
    def from_dataarray(cls, ref, da, dim='rec', child=Phases, child_kws=None, **kwargs):

        if child_kws is None:
            child_kws = {}

        items = []
        index = []
        for i, g in da.groupby(dim):
            index.append(i)
            items.append(child.from_dataarray(ref=ref, da=g, **child_kws))
        new = cls(items=items, index=index, **kwargs)

        # d = {}
        # for k in ['spinodals', 'binodals']:
        #     _k = '_' + k
        #     label = da.coords[k]
        #     features = np.unique(label[label > 0])
        #     for feature in features:
        #         idx = np.where(label == feature)[0][0]
        #         if _k not in d:
        #             d[_k] = [lnpis[idx]]
        #         else:
        #             d[_k].append(lnpis[idx])
        # for _k, v in d.items():
        #     if len(v) > 0:
        #         setattr(new, _k, v)
        return new



