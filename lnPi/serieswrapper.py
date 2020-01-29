"""
Alternative to ListIndex.  Will wrap stuff in Series
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
import xarray as xr

from .core import MaskedlnPi
from .extensions import AccessorMixin, ListAccessorMixin
from .cached_decorators import gcached

from .utils import (
    labels_to_masks, masks_to_labels, masks_change_convention,
    get_tqdm_build as get_tqdm, parallel_map_build as parallel_map)



class SeriesWrapper(AccessorMixin):
    """
    wrap object in series
    """
    def __init__(self,
                 data=None,
                 index=None,
                 dtype=None,
                 name=None,
                 base_class=None):

        if isinstance(data, self.__class__):
            x = data
            data = x.s

        self._base_class = base_class
        self._verify = self._base_class is not None

        series = pd.Series(data=data, index=index, dtype=dtype, name=name)
        self._verify_series(series)
        self._series = series
        self._cache = {}


    def _verify_series(self, series):
        if self._verify:
            for d in series:
                if not issubclass(type(d), self._base_class):
                    raise ValueError('all elements must be of type {}'.format(
                        self._base_class))
    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, series):
        self._cache = {}
        self._verify_series(series)
        self._series = series

    @property
    def s(self):
        return self.series

    def __iter__(self):
        return iter(self._series)

    def __next__(self):
        return next(self._series)

    @property
    def items(self):
        return self._series.values

    @property
    def index(self):
        return self._series.index

    @property
    def name(self):
        return self._series.name

    def copy(self):
        return self.__class__(data=self.s,
                              base_class=self._base_class)

    def new_like(self, data=None, index=None, **kwargs):
        return self.__class__(data=data,
                              index=index,
                              dtype=self.s.dtype,
                              name=self.s.name,
                              base_class=self._base_class,
                              **kwargs)

    def _wrapped_pandas_method(self, mtd, wrap=False, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(self._series, mtd)(*args, **kwargs)
        if wrap and type(val) == pd.Series:
            val = self.new_like(val)
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method('__getitem__', wrap=True, key=key)

    def __setitem__(self, idx, values):
        self._series[idx] = values

    def __repr__(self):
        return '<class {}>\n{}'.format(self.__class__.__name__, repr(self.s))

    def __str__(self):
        return str(self.s)

    def __len__(self):
        return len(self.s)

    def append(self, to_append, ignore_index=False, verify_integrity=False, inplace=False):
        if isinstance(to_append, self.__class__):
            to_append = to_append.series

        s = self._series.append(to_append,
                                ignore_index=ignore_index,
                                verify_integrity=verify_integrity)

        if inplace:
            self.series = s
        else:
            return self.new_like(s)

    def droplevel(self, level, axis=0):
        return self.new_like(self._series.droplevel(level=level, axis=axis))

    def apply(self, func, convert_dtype=True, args=(), wrap=False, **kwds):

        return self._wrapped_pandas_method('apply',
                                           wrap=wrap,
                                           func=func,
                                           convert_dtype=convert_dtype,
                                           args=args,
                                           **kwds)
    def sort_index(self, wrap=True, *args, **kwargs):
        return self._wrapped_pandas_method('sort_index', wrap=wrap, *args, **kwargs)

    def groupby(self,
                by=None,
                axis=0,
                level=None,
                as_index=True,
                sort=True,
                group_keys=True,
                squeeze=False,
                observed=False,
                wrap=False,
                **kwargs):
        """
        wrapper around groupby.

        Paremters
        ---------
        wrap : bool, default=False
            if True, try to wrap output in class of self

        See Also
        --------
        `pandas.Series.groupby` documentation
        """

        group = self.s.groupby(by=by,
                               axis=axis,
                               level=level,
                               as_index=as_index,
                               sort=sort,
                               group_keys=group_keys,
                               squeeze=squeeze,
                               observed=observed,
                               **kwargs)
        if wrap:
            return _Groupby(self, group)
        else:
            return group

    def groupby_allbut(self, drop, **kwargs):
        """
        groupby all but columns in drop
        """
        from .utils import allbut
        if not isinstance(drop, list):
            drop = [drop]
        by = allbut(self.index.names, *drop)
        return self.groupby(by=by, **kwargs)




    @classmethod
    def concat(cls, objs, concat_kws=None, *args, **kwargs):
        from collections.abc import Sequence, Mapping
        if isinstance(objs, Sequence):
            first = objs[0]
            if isinstance(first, cls):
                objs = (x._series for x in objs)
        elif isinstance(objs, Mapping):
            out = {}
            remap = None
            for k in objs:
                v = objs[k]
                if remap is None:
                    if isinstance(v, cls):
                        remap = True
                    else:
                        remap = False
                if remap:
                    out[k] = v._series
                else:
                    out[k] = v
            objs = out
        else:
            raise ValueError('bad input type {}'.format(type(first)))

        if concat_kws is None:
            concat_kws = {}

        s = pd.concat(objs, **concat_kws)
        return cls(s, *args, **kwargs)


# Accessors
class _CallableResult(object):
    def __init__(self, parent, series):
        self._parent = parent
        self._series = series

    def __call__(self, *args, **kwargs):
        return self._parent.new_like(self._series(*args, **kwargs))


class _Groupby(object):
    def __init__(self, parent, group):
        self._parent = parent
        self._group = group

    def __iter__(self):
        return ((meta, self._parent.new_like(x)) for meta, x in self._group)

    def __getattr__(self, attr):
        if hasattr(self._group, attr):
            out = getattr(self._group, attr)
            if callable(out):
                return _CallableResult(self._parent, out)
            else:
                return self._parent.new_like(out)
        else:
            raise AttributeError('no attribute {} in groupby'.format(attr))


@SeriesWrapper.decorate_accessor('loc')
class _LocIndexer(object):
    def __init__(self, parent):
        self._parent = parent
        self._loc = self._parent._series.loc

    def __getitem__(self, idx):
        out = self._loc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out

    def __setitem__(self, idx, values):
        self._parent._series.loc[idx] = values


@SeriesWrapper.decorate_accessor('iloc')
class _iLocIndexer(object):
    def __init__(self, parent):
        self._parent = parent
        self._iloc = self._parent._series.iloc

    def __getitem__(self, idx):
        out = self._iloc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out

    def __setitem__(self, idx, values):
        self._parent._series.iloc[idx] = values



@SeriesWrapper.decorate_accessor('query')
class _Query(object):
    def __init__(self, parent):
        self._parent = parent
        self._frame  = self._parent.index.to_frame().reset_index(drop=True)

    def __call__(self, expr, **kwargs):
        idx = self._frame.query(expr, **kwargs).index
        return self._parent.iloc[idx]




class CollectionlnPi(ListAccessorMixin, SeriesWrapper):
    _concat_dim = 'sample'
    _concat_coords = 'different'
    _use_joblib = True
    _xarray_output = True
    _xarray_unstack = True

    def __init__(self,
                 data,
                 index=None,
                 xarray_output=True,
                 concat_dim=None,
                 concat_coords=None,
                 unstack=True,
                 *args,
                 **kwargs):

        if concat_dim is not None:
            self._concat_dim = concat_dim
        if concat_coords is not None:
            self._concat_coords = concat_coords
        if xarray_output is not None:
            self._xarray_output = xarray_output
        if unstack is not None:
            self._xarray_unstack = unstack

        super(CollectionlnPi, self).__init__(data=data,
                                             index=index,
                                             *args,
                                             **kwargs)

        # update index name:
        #self._series.index.name = self._concat_dim

    def new_like(self, data=None, index=None):
        return super(CollectionlnPi,
                     self).new_like(data=data,
                                    index=index,
                                    concat_dim=self._concat_dim,
                                    concat_coords=self._concat_coords,
                                    xarray_output=self._xarray_output,
                                    unstack=self._xarray_unstack)


    def _verify_series(self, series):
        super(CollectionlnPi, self)._verify_series(series)
        if self._verify:
            state_kws = None
            for lnpi in series:
                if state_kws is None:
                    state_kws = lnpi.state_kws
                assert lnpi.state_kws == state_kws

    @property
    def state_kws(self):
        return self.iloc[0].state_kws

    @property
    def nlnz(self):
        """number of unique lnzs"""
        return len(self.index.droplevel('phase').drop_duplicates())


    def _get_lnz(self, component=None, iloc=0, zloc=None):
        """
        helper function to 
        returns self.iloc[idx].lnz[component]
        """
        if zloc is not None:
            s = self.zloc[zloc]._series
        else:
            s = self._series
        lnz = s.iloc[iloc].lnz
        if component is not None:
            lnz = lnz[component]
        return lnz


    def _get_level(self, level='phase'):
        """
        return level values from index
        """
        index = self.index
        if isinstance(index, pd.MultiIndex):
            level_idx = index.names.index(level)
            index = index.levels[level_idx]
        return index

    #@gcached()
    @property
    def _nrec(self):
        return len(self._series)

    @property
    def _lnpi_tot(self):
        return np.stack([x.filled() for x in self])

    @property
    def _lnpi_0_tot(self):
        return np.array([x.data.ravel()[0] for x in self])

    @property
    def _lnz_tot(self):
        return np.stack([x.lnz for x in self])

    def wrap_list_results(self, items):
        if self._xarray_output:
            x = items[0]
            if isinstance(x, xr.DataArray):
                return (xr.concat(items,
                                  self.index,
                                  coords=self._concat_coords))
            # else try to make array
        else:
            return items

    ##################################################
    # Constructors
    @classmethod
    def from_list(cls, items, index, *args, **kwargs):
        """
        create collection from list of lnPi's

        Parameters
        ----------
        items : sequence
            Sequence of lnPi
        index : sequence
            Sequence of phases ID for each lnPi
        *args, **kwargs :
            extra arguments to class constructor
        Returns
        -------
        output : class instance
        """
        df = pd.DataFrame(
            [lnpi._index_dict(phase) for lnpi, phase in zip(items, index)])
        index = pd.MultiIndex.from_frame(df)
        return cls(data=items, index=index, *args, **kwargs)


    @classmethod
    def from_builder(cls, lnzs, build_phases,
                     ref=None, build_phases_kws=None, nmax=None,
                     concat_kws=None, base_class=MaskedlnPi,
                     *args, **kwargs):
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
        Returns
        -------
        out : Collection object
        """
        if build_phases_kws is None:
            build_phases_kws = {}
        build_phases_kws = dict(build_phases_kws, phases_factory='None')
        seq = get_tqdm(lnzs, desc='build')
        L = parallel_map(build_phases, seq, ref=ref, nmax=nmax, **build_phases_kws)
        # return cls.concat(L, verify=verify, concat_kws=concat_kws, base_class=base_class,
        #                   *args, **kwargs)

        items = []
        index = []
        for data, idx in L:
            items += data
            index += list(idx)
        return cls.from_list(items, index, base_class=base_class, *args, **kwargs)



    ################################################################################
    # dataarray io
    def to_dataarray(self, dtype=np.uint8, reset_index=True, **kwargs):
        labels = []
        indexes = []

        for meta, g in self.groupby_allbut('phase'):
            indexes.append(g.index[[0]])

            features = np.array(g.index.get_level_values('phase')) + 1
            masks = [x.mask for x in g]
            labels.append(
                masks_to_labels(masks,
                                features=features,
                                convention=False,
                                dtype=dtype))

        index = indexes[0].append(indexes[1:])

        # df = self._series.reset_index(name='lnpi')
        # for meta, g in df.groupby(df.columns.drop(['phase', 'lnpi']).tolist()):

        #     indexes.append(g.drop(['phase', 'lnpi'], axis=1).iloc[0])

        #     features = []
        #     masks = []
        #     for _, gg in g.iterrows():
        #         phase = gg['phase']
        #         features.append(phase)
        #         masks.append(gg['lnpi'].mask)

        #     features = np.array(features) + 1
        #     labels.append(
        #         masks_to_labels(masks,
        #                         features=features,
        #                         convention=False,
        #                         dtype=dtype))
        # index = pd.MultiIndex.from_frame(pd.DataFrame(indexes))
        data = np.stack(labels)



        out = (
            xr.DataArray(
                data,
                dims=self.xgce.dims_rec + self.xgce.dims_n,
                name='labels',
            )
            .assign_coords(
                **{
                    self._concat_dim: index,
                    **self.state_kws
                })
            .assign_attrs(**self.xgce._standard_attrs)
        ) #yapf: disable

        if reset_index:
            out = out.reset_index(self._concat_dim)

        return out

    @classmethod
    def from_labels(cls,
                    ref,
                    labels,
                    lnzs,
                    features=None,
                    include_boundary=False,
                    labels_kws=None,
                    check_features=True,
                    **kwargs):

        if labels_kws is None:
            labels_kws = {}
        assert len(labels) == len(lnzs)

        items = []
        indexes = []

        for label, lnz in zip(labels, lnzs):
            lnpi = ref.reweight(lnz)

            masks, features_tmp = labels_to_masks(
                labels=label,
                features=features,
                include_boundary=include_boundary,
                convention=False,
                check_features=check_features,
                **labels_kws)

            index = list(np.array(features_tmp) - 1)
            items += lnpi.list_from_masks(masks, convention=False)
            indexes += index

        return cls.from_list(items=items, index=indexes, **kwargs)

    @classmethod
    def from_dataarray(cls,
                       ref,
                       da,
                       grouper='sample',
                       include_boundary=False,
                       labels_kws=None,
                       features=None,
                       check_features=True,
                       **kwargs):

        labels = []
        lnzs = []

        for i, g in da.groupby(grouper):
            lnzs.append(np.array([g.coords[k] for k in da.attrs['dims_lnz']]))
            labels.append(g.values)

        return cls.from_labels(
            ref=ref,
            labels=labels,
            lnzs=lnzs,
            labels_kws=labels_kws,
            features=features,
            include_boundary=include_boundary,
            check_features=check_features,
            **kwargs) # yapf: disable



################################################################################
# Accessors for ColleectionlnPi
@SeriesWrapper.decorate_accessor('zloc')
class _LocIndexer_unstack(object):
    """positional indexer for everything but phase"""
    def __init__(self, parent, level=['phase']):
        self._parent = parent
        self._level = level
        self._loc = self._parent._series.unstack(self._level).iloc

    def __getitem__(self, idx):
        out = self._loc[idx]
        if isinstance(out, pd.DataFrame):
            out = out.stack(self._level)
        else:
            out = out.dropna()

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out


