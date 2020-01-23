from __future__ import print_function, absolute_import, division

from collections import Iterable
from functools import partial

import numpy as np
import xarray as xr
import pandas as pd

from .cached_decorators import gcached, cached_clear
from .utils import labels_to_masks, masks_to_labels, masks_change_convention
from .utils import get_tqdm_build as get_tqdm
from .utils import parallel_map_build as parallel_map

from .extensions import AccessorMixin, ListAccessorMixin
from .extensions import decorate_listproperty, decorate_listaccessor


class _WrapListIndex(object):
    """
    wrap a 1-d list and index
    """

    def __init__(self, items, index=None, name=None):
        self._set_items(items)
        self._set_index(index, name)

    def _set_items(self, items):
        self._items = np.empty(len(items), dtype=np.object)
        if items is not None:
            self._items[:] = items

    def _set_index(self, index, name):
        if index is None:
            index = np.arange(len(self))
        self._name = name
        self._index = pd.Index(index, name=name)


    @property
    def items(self):
        return self._items

    @property
    def index(self):
        return self._index

    def copy(self):
        return self.__class__(self.items.tolist(), self.index.tolist(), self._index.name)

    def new_like(self , items, index):
        return self.__class__(items, index, self.index.name)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._items[idx]
        else:
            return self.new_like(self._items[idx], self._index[idx])

    def __setitem__(self, idx, values):
        self._items[idx] = values

    def __len__(self):
        return len(self._items)

    def append(self, value, index=None, get_index=True):
        if index is None:
            if get_index:
                try:
                    index = value.index
                except:
                    index = self.index[-1] + 1
            else:
                index = self.index[-1] + 1


        items = self.items.tolist() 
        items.append(value)

        idx = self.index.tolist()
        idx.append(index)

        self._set_items(items)
        self._set_index(idx, self.index.name)


    def extend(self, values, index=None):
        if index is None:
            assert isinstance(values, self.__class__)
            items = self.items.tolist() + values.items.tolist()
            idx   = self.index.tolist() + values.index.tolist()
        else:
            assert isinstance(values, list)
            assert isinstance(index, list)
            assert len(index) == len(values)
            items = self.items.tolist() + values
            idx   = self.index.tolist() + index

        self._set_items(items)
        self._set_index(idx, name=self.index.name)


    def __add__(self, values):
        assert isinstance(values, self.__class__)
        items = self.items.tolist() + values.items.tolist()
        idx   = self.index.tolist() + values.index.tolist()
        return self.new_like(items, idx)

    def __iadd__(self, values):
        if isinstance(values, self.__class__):
            self.extend(values)
        else:
            self.append(values)

    def as_series(self):
        return pd.Series(self.items, index=self.index)


class BaseCollection(AccessorMixin, ListAccessorMixin, _WrapListIndex):
    _CONCAT_DIM_ = None
    _CONCAT_COORDS_ = 'different'
    _USE_JOBLIB_ = False

    def __init__(self, items, index=None, xarray_output=True, concat_dim=None, concat_coords=None):
        if concat_dim is None:
            concat_dim = self._CONCAT_DIM_
        if concat_coords is None:
            concat_coords = self._CONCAT_COORDS_

        self._concat_dim = concat_dim
        self._concat_coords = concat_coords
        self._xarray_output = xarray_output
        super(BaseCollection, self).__init__(items, index, name=self._concat_dim)


    def copy(self):
        return self.__class__(
            items=self.items.tolist(),
            index=self.index.tolist(),
            xarray_output=self._xarray_output,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords)

    def new_like(self , items, index):
        if hasattr(index, 'tolist'):
            index = index.tolist()
        return self.__class__(
            items=items,
            index=index,
            xarray_output=self._xarray_output,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords)



    def wrap_list_results(self, items):
        if self._xarray_output:
            x = items[0]
            if isinstance(x, xr.DataArray):
                return (
                    xr.concat(items, self.index, coords=self._concat_coords)
                )
        # else try to make array
        try:
            return np.array(items)
        except:
            return items

    def __repr__(self):
        header = f"<{self.__class__.__name__}  ({self._concat_dim} : {len(self)})>"
        index = f"index {repr(self.index)}"
        return '\n'.join([header, index])


class BaseCollectionlnPi(BaseCollection):
    pass


class Phases(BaseCollectionlnPi):
    _CONCAT_DIM_ = 'phase'
    _CONCAT_COORDS_ = 'different'

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



def _cleanup_coords(da):
    out = da
    for k in da.attrs['dims_state'] + ['rec']:
        if 'phase' in da[k].dims:
            out = out.assign_coords(**{k : da[k].max('phase')})
    return out



class FlatCollectionlnPi(BaseCollectionlnPi):
    _CONCAT_DIM_ = 'sample'
    _CONCAT_COORDS_ = 'different'
    _USE_JOBLIB_ = True


    def __init__(self, items, index, parent_index_name,
                 unstack=True, set_index=None, xarray_output=True, concat_dim=None, concat_coords=None, clean_coords=True):
        # redo items and index
        items_flat = []
        index_flat = []

        for idx_c, phases in zip(index, items):
            for idx_p, phase in zip(phases.index, phases):
                items_flat.append(phase)
                index_flat.append((idx_c, idx_p))

        names = [parent_index_name, items[0].index.name]
        self._multiindex = pd.MultiIndex.from_tuples(index_flat, names=names)
        self._unstack = unstack
        self._xr_set_index = set_index
        self._clean_coords = clean_coords

        super(FlatCollectionlnPi, self).__init__(
            items=items_flat, index=None,
            xarray_output=xarray_output, concat_dim=concat_dim, concat_coords=concat_coords)


    def wrap_list_results(self, items):
        if self._xarray_output:
            x = items[0]
            if isinstance(x, xr.DataArray):
                out = (
                    xr.concat(items, self._concat_dim, coords=self._concat_coords)
                    .assign_coords(**{self._concat_dim : self._multiindex})
                )

                if self._xr_set_index is not None:
                    out = (
                        out
                        .reset_index(self._concat_dim)
                        .set_index(**{self._concat_dim : self._xr_set_index})
                    )

                if self._unstack:
                    out = out.unstack(self._concat_dim)

                    if self._clean_coords:
                        out = _cleanup_coords(out)

                return out

        # else try to make array
        try:
            return np.array(items)
        except:
            return items


@decorate_listproperty(['lnz'])
class CollectionPhases(BaseCollection):
    _CONCAT_DIM_ = 'rec'
    _CONCAT_COORDS_ = 'different'
    _USE_JOBLIB_ = True

    def __init__(self, items, index=None, xarray_output=True, concat_dim=None, concat_coords=None, unstack=True, set_index=None, clean_coords=True):
        self._unstack = unstack
        self._xr_set_index = set_index
        self._clean_coords = clean_coords

        super(CollectionPhases, self).__init__(
            items=items, index=index,
            xarray_output=xarray_output, concat_dim=concat_dim, concat_coords=concat_coords)



    def new_like(self , items, index, **kwargs):
        kwargs = dict(dict(
            xarray_output=self._xarray_output,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords,
            unstack=self._unstack,
            set_index=self._xr_set_index,
            clean_coords=self._clean_coords,
        ), **kwargs)

        if hasattr(index, 'tolist'):
            index = index.tolist()
        return self.__class__(
            items=items,
            index=index,
            **kwargs)



    @gcached()
    def flat(self):
        """
        flat indexer for calculations
        """
        return FlatCollectionlnPi(
            items=self.items, index=self.index, parent_index_name=self.index.name,
            unstack=self._unstack, set_index=self._xr_set_index, concat_dim='sample', concat_coords=self._concat_coords, clean_coords=self._clean_coords)


    def sort_by_lnz(self, comp=0, inplace=True, leave_index=True, **kwargs):
        order = np.argsort(self.lnz[:, comp])

        items = [self.items[i] for i in order]
        if leave_index:
            idx = self.index.tolist()
        else:
            idx   = [self.index[i] for i in order]
        if inplace:
            self._set_items(items)
            self._set_index(idx, name=self.index.name)
            self._cache = {}
        else:
            return self.new_like(items, idx)



    ##################################################
    #builders
    ##################################################
    @classmethod
    def from_builder(cls, lnzs, build_phases, ref=None, build_phases_kws=None, nmax=None, xarray_output=True, **kwargs):
        """
        build collection from scalar builder

        Parameters
        ----------
        lnzs : 1-D sequence
            lnz value for the varying value
        ref : lnpi_phases object
            lnpi_phases to reweight to get list of lnpi's
        build_phases : callable
            Typically one of `PhaseCreator.build_phases_mu` or `PhaseCreator.build_phases_dmu`
        build_phases_kws : optional
            optional arguments to `build_phases`
        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : Collection object
        """
        if build_phases_kws is None:
            build_phases_kws = {}
        seq = get_tqdm(lnzs, desc='build')
        L = parallel_map(build_phases, seq, ref=ref, nmax=nmax, **build_phases_kws)
        #L = [build_phases(lnz, ref=ref, nmax=nmax, **build_phases_kws) for lnz in seq]
        return cls(items=L, index=None, xarray_output=xarray_output, **kwargs)


    @classmethod
    def from_lnz_iter(cls, lnzs, ref=None, build_phases=None, build_phases_kws=None,  nmax=None, xarray_output=True, **kwargs):
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

        seq = get_tqdm(lnzs, desc='build')
        L = parallel_map(build_phases, seq, ref=ref, nmax=nmax, **build_phases_kws)
        #L = [build_phases(lnz=lnz, ref=ref, nmax=nmax, **build_phases_kws) for lnz in seq]
        return cls(items=L, index=None, xarray_output=xarray_output, **kwargs)

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






















# def _collapse_items(items, index, name):
#     new_items = []
#     new_index = []
#     for item, idx in zip(items, index):
#         pass



# class _FlatCollection(AccessorMixin, ListAccessorMixin):

#     _CONCAT_DIM_ = None
#     _CONCAT_COORDS_ = 'different'

#     def __init__(self, items, index=None, concat_dim=None, concat_coords=None, collapse=None):

#         if concat_dim is None:
#             concat_dim = self._CONCAT_DIM_
#         if concat_coords is None:
#             concat_coords = self._CONCAT_COORDS_
#         if collapse is None:
#             collapse = self._COLLAPSE_

#         self._concat_dim = concat_dim
#         self._concat_coords = concat_coords
#         self._collapse = collapse

