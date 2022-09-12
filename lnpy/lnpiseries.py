"""
Alternative to ListIndex.  Will wrap stuff in Series
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import xarray as xr

from .cached_decorators import gcached
from .extensions import AccessorMixin
from .utils import get_tqdm_build as get_tqdm
from .utils import labels_to_masks, masks_to_labels
from .utils import parallel_map_build as parallel_map


class SeriesWrapper(AccessorMixin):
    """
    wrap object in series
    """

    def __init__(
        self, data=None, index=None, dtype=None, name=None, base_class="first"
    ):

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
            base_class = self._base_class
            if isinstance(base_class, str) and base_class.lower() == "first":
                base_class = type(series.iloc[0])
            for d in series:
                if not issubclass(type(d), base_class):
                    raise ValueError(
                        "all elements must be of type {}".format(base_class)
                    )

    @property
    def series(self):
        """View of the underlying :class:`pandas.Series`"""
        return self._series

    @series.setter
    def series(self, series):
        self._cache = {}
        self._verify_series(series)
        self._series = series

    @property
    def s(self):
        """Alias to `self.series`"""
        return self.series

    def __iter__(self):
        return iter(self._series)

    def __next__(self):
        return next(self._series)

    @property
    def values(self):
        """Series values"""
        return self._series.values

    @property
    def items(self):
        """Alias to `self.values`"""
        return self.values

    @property
    def index(self):
        """Series index"""
        return self._series.index

    @property
    def name(self):
        """Series name"""
        return self._series.name

    def copy(self):
        return type(self)(data=self.s, base_class=self._base_class)

    def new_like(self, data=None, index=None, **kwargs):
        """Create new object with optional new data/index"""
        return self.__class__(
            data=data,
            index=index,
            dtype=self.s.dtype,
            name=self.s.name,
            base_class=self._base_class,
            **kwargs
        )

    def _wrapped_pandas_method(self, mtd, wrap=False, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(self._series, mtd)(*args, **kwargs)
        if wrap and type(val) == pd.Series:
            val = self.new_like(val)
        return val

    def __getitem__(self, key):
        """Interface to :meth:`pandas.Series.__getitem__`"""
        return self._wrapped_pandas_method("__getitem__", wrap=True, key=key)

    def xs(self, key, axis=0, level=None, drop_level=False, wrap=True):
        """Interface to :meth:`pandas.Series.xs`"""
        return self._wrapped_pandas_method(
            "xs", wrap=wrap, key=key, axis=axis, level=level, drop_level=drop_level
        )

    def __setitem__(self, idx, values):
        """Interface to :meth:`pandas.Series.__setitem__`"""
        self._series[idx] = values

    def __repr__(self):
        return "<class {}>\n{}".format(self.__class__.__name__, repr(self.s))

    def __str__(self):
        return str(self.s)

    def __len__(self):
        return len(self.s)

    def append(
        self,
        to_append,
        ignore_index=False,
        verify_integrity=True,
        concat_kws=None,
        inplace=False,
    ):
        """Interface to :meth:`pandas.Series.append`


        Parameters
        ----------
        to_append : object
            Object to append
        ignore_index : bool, default=False
        verify_integrity : bool, default=True
        concat_kws : mapping, optional
            Extra arguments to
        inplace : bool, default = False

        See Also
        --------
        pandas.Series.append
        """
        if isinstance(to_append, self.__class__):
            to_append = to_append.series

        if concat_kws is None:
            concat_kws = {}

        s = pd.concat(
            (self.series, to_append),
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            **concat_kws
        )

        # s = self._series.append(
        #     to_append, ignore_index=ignore_index, verify_integrity=verify_integrity
        # )

        if inplace:
            self.series = s
        else:
            return self.new_like(s)

    def droplevel(self, level, axis=0):
        """New object with dropped level

        See Also
        --------
        pandas.Series.droplevel
        """
        return self.new_like(self._series.droplevel(level=level, axis=axis))

    def apply(self, func, convert_dtype=True, args=(), wrap=False, **kwds):
        """Interface to :meth:`pandas.Series.apply`"""

        return self._wrapped_pandas_method(
            "apply",
            wrap=wrap,
            func=func,
            convert_dtype=convert_dtype,
            args=args,
            **kwds
        )

    def sort_index(self, wrap=True, *args, **kwargs):
        """Interface to :meth:`pandas.Series.sort_index`"""
        return self._wrapped_pandas_method("sort_index", wrap=wrap, *args, **kwargs)

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        # squeeze=False,
        observed=False,
        wrap=False,
        **kwargs
    ):
        """
        Wrapper around :meth:`pandas.Series.groupby`.

        Paremters
        ---------
        wrap : bool, default=False
            if True, try to wrap output in class of self

        See Also
        --------
        pandas.Series.groupby
        """

        group = self.s.groupby(
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            # squeeze=squeeze,
            observed=observed,
            **kwargs
        )
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
    def _concat_to_series(cls, objs, **concat_kws):
        from collections.abc import Mapping, Sequence

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
            raise ValueError("bad input type {}".format(type(first)))
        return pd.concat(objs, **concat_kws)

    def concat_like(self, objs, **concat_kws):
        """Concat a sequence of objects like `self`"""
        s = self._concat_to_series(objs, **concat_kws)
        return self.new_like(s)

    @classmethod
    def concat(cls, objs, concat_kws=None, *args, **kwargs):
        """Create collection from sequence of objects"""
        if concat_kws is None:
            concat_kws = {}
        s = cls._concat_to_series(objs, **concat_kws)
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
            raise AttributeError("no attribute {} in groupby".format(attr))


@SeriesWrapper.decorate_accessor("loc")
class _LocIndexer(object):
    """
    Indexer by value.

    See :attr:`pandas.Series.loc`
    """

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


@SeriesWrapper.decorate_accessor("iloc")
class _iLocIndexer(object):
    """Indexer by position.

    See :attr:`pandas.Series.iloc`
    """

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


@SeriesWrapper.decorate_accessor("query")
class _Query(object):
    """
    Select values by string query.

    See :meth:`pandas.DataFrame.query`
    """

    def __init__(self, parent):
        self._parent = parent
        self._frame = self._parent.index.to_frame().reset_index(drop=True)

    def __call__(self, expr, **kwargs):
        idx = self._frame.query(expr, **kwargs).index
        return self._parent.iloc[idx]


class lnPiCollection(SeriesWrapper):
    # class lnPiCollection:
    r"""
    Wrapper around :class:`pandas.Series` for collection of :class:`lnpy.lnPiMasked` objects.


    Parameters
    ----------
    data : sequence of lnPiMasked
        :math:`\ln \Pi(N)` instances to consider.
    index : array-like, pandas.Index, pandas.MultiIndex, optional
        Index to apply to Series.
    xarray_output : bool, default = True
        If True, then wrap lnPiCollection outputs in :class:`~xarray.DataArray`
    concat_dim : str, optional
        Name of dimensions to concat results along.
        Also Used by :class:`~lnpy.ensembles.xGrandCanonical`.
    concat_coords : string, optional
        parameters `coords `to :func:`xarray.concat`
    unstack : bool, default=True
        If True, then outputs will be unstacked using :meth:`xarray.DataArray.unstack`
    *args **kwargs
        Extra arguments to Series constructor

    """

    _concat_dim = "sample"
    _concat_coords = "different"
    _use_joblib = True
    _xarray_output = True
    _xarray_unstack = True
    _xarray_dot_kws = {"optimize": "optimal"}
    _use_cache = True

    def __init__(
        self,
        data,
        index=None,
        xarray_output=True,
        concat_dim=None,
        concat_coords=None,
        unstack=True,
        *args,
        **kwargs
    ):

        if concat_dim is not None:
            self._concat_dim = concat_dim
        if concat_coords is not None:
            self._concat_coords = concat_coords
        if xarray_output is not None:
            self._xarray_output = xarray_output
        if unstack is not None:
            self._xarray_unstack = unstack

        super().__init__(data=data, index=index, *args, **kwargs)

        # update index name:
        # self._series.index.name = self._concat_dim

    def new_like(self, data=None, index=None):
        """Create new object with optional new data/index."""
        return super().new_like(
            data=data,
            index=index,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords,
            xarray_output=self._xarray_output,
            unstack=self._xarray_unstack,
        )

    def _verify_series(self, series):
        super()._verify_series(series)
        if self._verify:
            first = series.iloc[0]
            state_kws = first.state_kws
            shape = first.shape
            # _base  = first._base

            for lnpi in series:
                assert lnpi.state_kws == state_kws
                assert lnpi.shape == shape
                # would like to do this, but
                # fails for parallel builds
                # assert lnpi._base is _base

    # repr
    @gcached()
    def _lnz_series(self):
        return self._series.apply(lambda x: x.lnz)

    def __repr__(self):
        return "<class {}>\n{}".format(self.__class__.__name__, repr(self._lnz_series))

    def __str__(self):
        return str(self._lnz_series)

    @property
    def state_kws(self):
        """state_kws from first :class:`~lnpy.lnPiMasked`"""
        return self.iloc[0].state_kws

    @property
    def nlnz(self):
        """Number of unique lnzs"""
        return len(self.index.droplevel("phase").drop_duplicates())

    @gcached()
    def index_frame(self):
        """
        DataFrame of values for each sample

        includes a column 'lnz_index' which is the unique lnz values
        regardless of phase
        """
        sample_frame = (
            self.index.droplevel("phase")
            .drop_duplicates()
            .to_frame()
            .assign(lnz_sample=lambda x: np.arange(len(x)))["lnz_sample"]
        )
        index_frame = (
            self.index.to_frame()
            .reset_index("phase", drop=True)[["phase"]]
            .assign(lnz_index=lambda x: sample_frame[x.index])
            .reset_index()
        )
        return index_frame

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

    def _get_level(self, level="phase"):
        """
        return level values from index
        """
        index = self.index
        if isinstance(index, pd.MultiIndex):
            level_idx = index.names.index(level)
            index = index.levels[level_idx]
        return index

    def get_index_level(self, level="phase"):
        """Get index values for specified level"""
        return self._get_level(level=level)

    # @gcached()
    @property
    def _nrec(self):
        return len(self._series)

    def _lnpi_tot(self, fill_value=None):
        # old method
        # return np.stack([x.filled() for x in self])

        # new method
        # this is no faster than the original
        # but makes clear where the time is being spent
        first = self.iloc[0]
        n = len(self)
        out = np.empty((n,) + first.shape, dtype=first.dtype)
        seq = get_tqdm((x.filled(fill_value) for x in self), total=n)
        for i, x in enumerate(seq):
            out[i, ...] = x
        return out

    # @property
    # def _lnpi_0_tot(self):
    #     return np.array([x.data.ravel()[0] for x in self])

    @property
    def _lnz_tot(self):
        return np.stack([x.lnz for x in self])

    def _pi_params(self, fill_value=None):
        first = self.iloc[0]
        n = len(self)

        pi_norm = np.empty((n,) + first.shape, dtype=first.dtype)
        pi_sum = np.empty(n, dtype=first.dtype)
        lnpi_zero = np.empty(n, dtype=first.dtype)

        seq = get_tqdm(
            (x._pi_params(fill_value) for x in self), total=n, desc="pi_norm"
        )

        for i, (_pi_norm, _pi_sum, _lnpi_zero) in enumerate(seq):
            pi_norm[i, ...] = _pi_norm
            pi_sum[i, ...] = _pi_sum
            lnpi_zero[i, ...] = _lnpi_zero
        return pi_norm, pi_sum, lnpi_zero

    def wrap_list_results(self, items):
        """Unitility to wrap output in :class:xarray.DataArray"""
        if self._xarray_output:
            x = items[0]
            if isinstance(x, xr.DataArray):
                return xr.concat(items, self.index, coords=self._concat_coords)
            # else try to make array
        else:
            return items

    ##################################################
    # Constructors
    @classmethod
    def from_list(cls, items, index, *args, **kwargs):
        """
        Create collection from list of :class:`lnpy.lnPiMasked` objects.

        Parameters
        ----------
        items : sequence of lnPiMasked
            Sequence of lnPi
        index : sequence
            Sequence of phases ID for each lnPi
        *args
            Extra positional arguments to `cls`
        **kwargs :
            Extra keyword arguments to `cls`
        Returns
        -------
        output : class instance
        """
        df = pd.DataFrame(
            [lnpi._index_dict(phase) for lnpi, phase in zip(items, index)]
        )
        index = pd.MultiIndex.from_frame(df)
        return cls(data=items, index=index, *args, **kwargs)

    @classmethod
    def from_builder(
        cls,
        lnzs,
        build_phases,
        ref=None,
        build_kws=None,
        nmax=None,
        concat_kws=None,
        base_class="first",
        *args,
        **kwargs
    ):
        """
        Build collection from scalar builder

        Parameters
        ----------
        lnzs : 1-D sequence
            lnz value for the varying value
        ref : lnpi_phases object
            lnpi_phases to reweight to get list of lnpi's
        build_phases : callable
            Typically one of `PhaseCreator.build_phases_mu` or `PhaseCreator.build_phases_dmu`
        build_kws : optional
            optional arguments to `build_phases`
        Returns
        -------
        out : Collection object


        See Also
        --------
        ~lnpy.segment.Segmenter.build_phases
        ~lnpy.segment.Segmenter.build_phases_mu
        ~lnpy.segment.Segmenter.build_phases_dmu

        """
        if build_kws is None:
            build_kws = {}
        build_kws = dict(build_kws, phases_factory="None")
        seq = get_tqdm(lnzs, desc="build")
        L = parallel_map(build_phases, seq, ref=ref, nmax=nmax, **build_kws)
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
        """
        Convert collection to a :class:`~xarray.DataArray`
        """
        labels = []
        indexes = []

        for meta, g in self.groupby_allbut("phase"):
            indexes.append(g.index[[0]])

            features = np.array(g.index.get_level_values("phase")) + 1
            masks = [x.mask for x in g]
            labels.append(
                masks_to_labels(masks, features=features, convention=False, dtype=dtype)
            )

        index = indexes[0].append(indexes[1:])

        data = np.stack(labels)

        out = (
            xr.DataArray(
                data,
                dims=self.xge.dims_rec + self.xge.dims_n,
                name="labels",
            )
            .assign_coords(**{self._concat_dim: index, **self.state_kws})
            .assign_attrs(**self.xge._standard_attrs)
        )  # yapf: disable

        if reset_index:
            out = out.reset_index(self._concat_dim)

        return out

    @classmethod
    def from_labels(
        cls,
        ref,
        labels,
        lnzs,
        features=None,
        include_boundary=False,
        labels_kws=None,
        check_features=True,
        **kwargs
    ):
        r"""
        Create from reference :class:`~lnpy.lnPiMasked` and labels array


        Parameters
        ----------
        ref : lnPiMasked
        labels : sequence of label arrays
            Each `labels[i]` will be used to construct multiple phases from single
            (reweighted)  :math:`ln \Pi(N)`
        lnzs : sequence
            Each lnzs[i] will be passed to ``ref.reweight``.
        features : int, optional
        include_boundary : bool, default=False
        labels_kws : mapping, optional
        check_features : bool, default = True
        **kwargs
            Extra arguments past to :meth:`from_list`

        See Also
        --------
        labels_to_masks
        from_list
        """

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
                **labels_kws
            )

            index = list(np.array(features_tmp) - 1)
            items += lnpi.list_from_masks(masks, convention=False)
            indexes += index

        return cls.from_list(items=items, index=indexes, **kwargs)

    @classmethod
    def from_dataarray(
        cls,
        ref,
        da,
        grouper="sample",
        include_boundary=False,
        labels_kws=None,
        features=None,
        check_features=True,
        **kwargs
    ):
        """
        Create a collection from DataArray of labels

        Parameters
        ----------
        ref : lnPiMasked
        da : DataArray or labels
        grouper : Hashable
            Name of dimension(s) to group along to give a single label array


        See Also
        --------
        from_labels


        """

        labels = []
        lnzs = []

        for i, g in da.groupby(grouper):
            lnzs.append(np.array([g.coords[k] for k in da.attrs["dims_lnz"]]))
            labels.append(g.values)

        return cls.from_labels(
            ref=ref,
            labels=labels,
            lnzs=lnzs,
            labels_kws=labels_kws,
            features=features,
            include_boundary=include_boundary,
            check_features=check_features,
            **kwargs
        )  # yapf: disable


################################################################################
# Accessors for ColleectionlnPi
@SeriesWrapper.decorate_accessor("zloc")
class _LocIndexer_unstack_zloc(object):
    """positional indexer for everything but phase"""

    def __init__(self, parent, level=["phase"]):
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


@SeriesWrapper.decorate_accessor("mloc")
class _LocIndexer_unstack_mloc(object):
    """indexer with pandas index"""

    def __init__(self, parent, level=["phase"]):
        self._parent = parent
        self._level = level
        self._index = self._parent.index

        self._index_names = set(self._index.names)
        self._loc = self._parent._series.iloc

    def _get_loc_idx(self, idx):
        index = self._index
        if isinstance(idx, pd.MultiIndex):
            # names in idx and
            drop = list(self._index_names - set(idx.names))
            index = index.droplevel(drop)
            # reorder idx
            idx = idx.reorder_levels(index.names)
        else:
            drop = list(set(index.names) - {idx.name})
            index = index.droplevel(drop)
        indexer = index.get_indexer_for(idx)
        return indexer

    def __getitem__(self, idx):
        indexer = self._get_loc_idx(idx)
        out = self._loc[indexer]

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out


# play around with sample index
# lnz_0 = np.random.rand(20)

# L = []
# for z in lnz_0:
#     repeat = np.random.randint(1, 4)
#     L += [(z, i) for i in range(repeat)]

# # example dataset
# idx = pd.MultiIndex.from_tuples(L, names=['lnz_0','phase'])
# s = pd.Series(range(len(L)), index=idx)

# def get_sample_index(s):
#     # get mapping from row values to sample
#     idx = s.index
#     idx_less_phase = idx.droplevel('phase')
#     idx_less_phase_unique = idx_less_phase.drop_duplicates()

#     mapping = pd.Series(range(len(idx_less_phase_unique)), idx_less_phase_unique)

#     # # sample values
#     # samples = mapping.loc[idx_less_phase].values

#     # make new index
#     idx_samp = (
#     idx
#     .to_frame()
#     .assign(sample=mapping.loc[idx_less_phase].values)
#     .set_index(['sample'] + idx.names)
#     .index
#     )
#     return idx_samp
