"""
Collection of lnPi objects (:mod:`~lnpy.lnpiseries`)
====================================================
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Generic, overload
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from module_utilities import cached

from ._typing import T_Element, T_SeriesWrapper
from .docstrings import docfiller
from .extensions import AccessorMixin
from .lnpidata import lnPiMasked

# lazy loads
from .utils import get_tqdm_build as get_tqdm
from .utils import labels_to_masks, masks_to_labels
from .utils import parallel_map_build as parallel_map

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Hashable,
        Iterable,
        Iterator,
        Literal,
        Mapping,
        Sequence,
    )

    from numpy.typing import ArrayLike, DTypeLike
    from pandas.core.groupby.generic import SeriesGroupBy
    from typing_extensions import Self

    from . import ensembles, lnpienergy, stability
    from ._typing import IndexingInt, MyNDArray, Scalar


class SeriesWrapper(AccessorMixin, Generic[T_Element]):
    """wrap object in series"""

    def __init__(
        self,
        data: Sequence[T_Element] | pd.Series[Any],
        index: ArrayLike | pd.Index[Any] | pd.MultiIndex | None = None,
        dtype: DTypeLike | None = None,
        name: Hashable | None = None,
        base_class: str | type = "first",
    ) -> None:
        if isinstance(data, self.__class__):
            x = data
            data = x.s

        self._base_class = base_class
        self._verify = self._base_class is not None

        series = pd.Series(data=data, index=index, dtype=dtype, name=name)  # type: ignore
        self._verify_series(series)
        self._series = series
        self._cache: dict[str, Any] = {}

    def _verify_series(self, series: pd.Series[Any]) -> None:
        if self._verify:
            base_class = self._base_class
            if isinstance(base_class, str) and base_class.lower() == "first":
                base_class = type(series.iloc[0])
            assert not isinstance(base_class, str)

            for d in series:
                if not issubclass(type(d), base_class):
                    raise ValueError(f"all elements must be of type {base_class}")

    @property
    def series(self) -> pd.Series[Any]:
        """View of the underlying :class:`pandas.Series`"""
        return self._series

    @series.setter
    def series(self, series: pd.Series[Any]) -> None:
        self._cache = {}
        self._verify_series(series)
        self._series = series

    @property
    def s(self) -> pd.Series[Any]:
        """Alias to :meth:`series`"""
        return self.series

    def __iter__(self) -> Iterator[T_Element]:
        return iter(self._series)

    @property
    def values(self) -> MyNDArray:
        """Series values"""
        return self._series.values  # type: ignore

    @property
    def items(self) -> MyNDArray:
        """Alias to :attr:`values`"""
        return self.values

    @property
    def index(self) -> pd.Index[Any]:
        """Series index"""
        return self._series.index

    @property
    def name(self) -> Hashable:
        """Series name"""
        return self._series.name

    def copy(self) -> Self:
        return type(self)(data=self.s, base_class=self._base_class)

    def new_like(
        self,
        data: Sequence[T_Element] | pd.Series[Any] | None = None,
        index: ArrayLike | pd.Index[Any] | pd.MultiIndex | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create new object with optional new data/index"""
        if data is None:
            data = self.s

        return type(self)(
            data=data,
            index=index,
            dtype=self.s.dtype,  # type: ignore
            name=self.s.name,
            base_class=self._base_class,
            **kwargs,
        )

    def _wrapped_pandas_method(
        self, mtd: str, wrap: bool = False, *args: Any, **kwargs: Any
    ) -> T_Element | pd.Series[Any] | Self:
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(self._series, mtd)(*args, **kwargs)
        if wrap and type(val) == pd.Series:
            val = self.new_like(val)
        return val  # type: ignore

    def __getitem__(self, key: Any) -> Self | T_Element:
        """Interface to :meth:`pandas.Series.__getitem__`"""
        return self._wrapped_pandas_method("__getitem__", wrap=True, key=key)  # type: ignore

    def xs(
        self,
        key: Hashable | Sequence[Hashable],
        axis: int = 0,
        level: Hashable | Sequence[Hashable] | None = None,
        drop_level: bool = False,
        wrap: bool = True,
    ) -> Self | pd.Series[Any] | T_Element:
        """Interface to :meth:`pandas.Series.xs`"""
        return self._wrapped_pandas_method(
            "xs", wrap=wrap, key=key, axis=axis, level=level, drop_level=drop_level
        )

    def __setitem__(
        self, idx: Any, values: T_Element | Sequence[T_Element] | pd.Series[Any]
    ) -> None:
        """Interface to :meth:`pandas.Series.__setitem__`"""
        self._series[idx] = values

    def __repr__(self) -> str:
        return f"<class {self.__class__.__name__}>\n{repr(self.s)}"

    def __str__(self) -> str:
        return str(self.s)

    def __len__(self) -> int:
        return len(self.s)

    def append(
        self,
        to_append: pd.Series[Any] | Self,
        ignore_index: bool = False,
        verify_integrity: bool = True,
        concat_kws: Mapping[str, Any] | None = None,
        inplace: bool = False,
    ) -> Self:
        """
        Interface to :func:`pandas.concat`


        Parameters
        ----------
        to_append : object
            Object to append
        ignore_index : bool, default=False
        verify_integrity : bool, default=True
        concat_kws : mapping, optional
            Extra arguments to
        inplace : bool, default=False

        See Also
        --------
        pandas.concat
        """
        if hasattr(to_append, "series") and isinstance(to_append.series, pd.Series):
            series = to_append.series
        elif isinstance(to_append, pd.Series):
            series = to_append
        else:
            raise ValueError(f"Unknown to append type={type(to_append)}")

        if concat_kws is None:
            concat_kws = {}

        s = pd.concat(
            (self.series, series),
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            **concat_kws,
        )

        if inplace:
            self.series = s
            return self
        else:
            return self.new_like(s)

    def droplevel(self, level: int | Hashable | Sequence[int | Hashable]) -> Self:
        """
        New object with dropped level

        See Also
        --------
        pandas.Series.droplevel
        """
        return self.new_like(self._series.droplevel(level=level, axis=0))  # type: ignore

    def apply(
        self,
        func: Callable[..., Any],
        convert_dtype: bool = True,
        args: tuple[Any, ...] = (),
        wrap: bool = False,
        **kwds: Any,
    ) -> Self | pd.Series[Any]:
        """Interface to :meth:`pandas.Series.apply`"""

        return self._wrapped_pandas_method(  # type: ignore
            "apply",
            wrap=wrap,
            func=func,
            convert_dtype=convert_dtype,
            args=args,
            **kwds,
        )

    def sort_index(self, *args: Any, **kwargs: Any) -> Self:
        """Interface to :meth:`pandas.Series.sort_index`"""
        return self._wrapped_pandas_method("sort_index", *args, wrap=True, **kwargs)  # type: ignore

    @overload
    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = ...,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        observed: bool = ...,
        dropna: bool = ...,
        wrap: Literal[False] = ...,
    ) -> SeriesGroupBy[Any, Any]:
        ...

    @overload
    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = ...,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        observed: bool = ...,
        dropna: bool = ...,
        wrap: Literal[True],
    ) -> _Groupby[Self, T_Element]:
        ...

    @overload
    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = ...,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        observed: bool = ...,
        dropna: bool = ...,
        wrap: bool,
    ) -> SeriesGroupBy[Any, Any] | _Groupby[Self, T_Element]:
        ...

    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = None,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        # squeeze=False,
        observed: bool = False,
        dropna: bool = True,
        wrap: bool = False,
    ) -> SeriesGroupBy[Any, Any] | _Groupby[Self, T_Element]:
        """
        Wrapper around :meth:`pandas.Series.groupby`.

        Parameters
        ----------
        wrap : bool, default=False
            if True, try to wrap output in class of self

        See Also
        --------
        pandas.Series.groupby
        """
        # TODO: fix types
        group = self.s.groupby(  # type: ignore
            by=by,
            axis=0,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            # squeeze=squeeze,
            observed=observed,
            dropna=dropna,
        )

        if wrap:
            return _Groupby(self, group)
        else:
            return group  # type: ignore

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: Literal[False] = ...,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any]:
        ...

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: Literal[True],
        **kwargs: Any,
    ) -> _Groupby[Self, T_Element]:
        ...

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: bool,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any] | _Groupby[Self, T_Element]:
        ...

    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: bool = False,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any] | _Groupby[Self, T_Element]:
        """Groupby all but columns in drop"""
        from .utils import allbut

        if isinstance(drop, list):
            drop = tuple(drop)
        elif not isinstance(drop, tuple):
            drop = (drop,)

        by = allbut(self.index.names, *drop)

        # To suppress annoying errors.
        if len(by) == 1:
            return self.groupby(by=by[0], wrap=wrap, **kwargs)
        else:
            return self.groupby(by=by, wrap=wrap, **kwargs)

    @classmethod
    def _concat_to_series(
        cls,
        objs: Sequence[Self]
        | Sequence[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        **concat_kws: Any,
    ) -> pd.Series[Any]:
        from collections.abc import Mapping, Sequence

        if isinstance(objs, Sequence):
            first = objs[0]
            if isinstance(first, cls):
                objs = tuple(x._series for x in objs)
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
            raise ValueError(f"bad input type {type(objs[0])}")
        return pd.concat(objs, **concat_kws)  # type: ignore

    def concat_like(
        self,
        objs: Sequence[Self]
        | Sequence[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        **concat_kws: Any,
    ) -> Self:
        """Concat a sequence of objects like `self`"""
        s = self._concat_to_series(objs, **concat_kws)
        return self.new_like(s)

    @classmethod
    def concat(
        cls,
        objs: Sequence[Self]
        | Sequence[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        concat_kws: Mapping[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create collection from sequence of objects"""
        if concat_kws is None:
            concat_kws = {}
        s = cls._concat_to_series(objs, **concat_kws)
        return cls(s, *args, **kwargs)

    # Note: use property(cached.meth(func)) here
    # normal cache.prop has some logic issues
    # with this pattern.
    @property
    @cached.meth
    def loc(self) -> _LocIndexer[Self, T_Element]:
        return _LocIndexer(self)

    @property
    @cached.meth
    def iloc(self) -> _iLocIndexer[Self, T_Element]:
        return _iLocIndexer(self)

    @property
    @cached.meth
    def query(self) -> _Query[Self, T_Element]:
        return _Query(self)

    @property
    @cached.meth
    def zloc(self) -> _LocIndexer_unstack_zloc[Self, T_Element]:
        return _LocIndexer_unstack_zloc(self)

    @property
    @cached.meth
    def mloc(self) -> _LocIndexer_unstack_mloc[Self, T_Element]:
        return _LocIndexer_unstack_mloc(self)


# Accessors
class _CallableResult(Generic[T_SeriesWrapper, T_Element]):
    def __init__(self, parent: T_SeriesWrapper, func: Callable[..., Any]) -> None:
        functools.update_wrapper(self, func)

        self._parent = parent
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> T_SeriesWrapper:
        return self._parent.new_like(self._func(*args, **kwargs))


class _Groupby(Generic[T_SeriesWrapper, T_Element]):
    def __init__(self, parent: T_SeriesWrapper, group: SeriesGroupBy[Any, Any]) -> None:
        self._parent = parent
        self._group = group

    def __iter__(self) -> Iterator[tuple[Any, T_SeriesWrapper]]:
        return ((meta, self._parent.new_like(x)) for meta, x in self._group)

    def __getattr__(
        self, attr: str
    ) -> _CallableResult[T_SeriesWrapper, T_Element] | T_SeriesWrapper:
        if hasattr(self._group, attr):
            out = getattr(self._group, attr)
            if callable(out):
                return _CallableResult(self._parent, out)
            else:
                return self._parent.new_like(out)
        else:
            raise AttributeError(f"no attribute {attr} in groupby")


# @SeriesWrapper.decorate_accessor("loc")
class _LocIndexer(Generic[T_SeriesWrapper, T_Element]):
    """
    Indexer by value.

    See :attr:`pandas.Series.loc`
    """

    def __init__(self, parent: T_SeriesWrapper) -> None:
        self._parent = parent
        self._loc = self._parent._series.loc

    @overload
    def __getitem__(self, idx: Scalar | tuple[Scalar, ...]) -> T_Element:
        ...

    @overload
    def __getitem__(
        self,
        idx: Sequence[Scalar] | pd.Index[Any] | slice | Callable[[pd.Series[Any]], Any],
    ) -> T_SeriesWrapper:
        ...

    @overload
    def __getitem__(self, idx: Any) -> T_Element | T_SeriesWrapper:
        ...

    def __getitem__(self, idx: Any) -> T_Element | T_SeriesWrapper:
        out = self._loc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out  # type: ignore

    def __setitem__(
        self, idx: Any, values: T_Element | pd.Series[Any] | Sequence[T_Element]
    ) -> None:
        self._parent._series.loc[idx] = values


# @SeriesWrapper.decorate_accessor("iloc")
class _iLocIndexer(Generic[T_SeriesWrapper, T_Element]):
    """
    Indexer by position.

    See :attr:`pandas.Series.iloc`
    """

    def __init__(self, parent: T_SeriesWrapper) -> None:
        self._parent = parent
        self._iloc = self._parent._series.iloc

    @overload
    def __getitem__(self, idx: IndexingInt) -> T_Element:
        ...

    @overload
    def __getitem__(
        self, idx: Sequence[int] | pd.Index[Any] | slice
    ) -> T_SeriesWrapper:
        ...

    @overload
    def __getitem__(self, idx: Any) -> T_Element | T_SeriesWrapper:
        ...

    def __getitem__(self, idx: Any) -> T_Element | T_SeriesWrapper:
        out = self._iloc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out  # type: ignore

    def __setitem__(
        self, idx: Any, values: T_Element | pd.Series[Any] | Sequence[T_Element]
    ) -> None:
        self._parent._series.iloc[idx] = values


# @SeriesWrapper.decorate_accessor("query")
class _Query(Generic[T_SeriesWrapper, T_Element]):
    """
    Select values by string query.

    See :meth:`pandas.DataFrame.query`
    """

    def __init__(self, parent: T_SeriesWrapper) -> None:
        self._parent = parent
        self._frame: pd.DataFrame = self._parent.index.to_frame().reset_index(drop=True)

    def __call__(self, expr: str, **kwargs: Any) -> T_SeriesWrapper:
        idx = self._frame.query(expr, **kwargs).index
        return self._parent.iloc[idx]  # type: ignore


# @SeriesWrapper.decorate_accessor("zloc")
class _LocIndexer_unstack_zloc(Generic[T_SeriesWrapper, T_Element]):
    """positional indexer for everything but phase"""

    def __init__(
        self,
        parent: T_SeriesWrapper,
        level: Hashable | Sequence[Hashable] = ("phase",),
    ) -> None:
        self._parent = parent
        self._level = level
        self._loc = self._parent._series.unstack(self._level).iloc

    def __getitem__(self, idx: Any) -> T_SeriesWrapper:
        out = self._loc[idx]
        if isinstance(out, pd.DataFrame):
            out = out.stack(self._level)
        else:
            out = out.dropna()

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        else:
            raise ValueError("unknown indexer for zloc")
        return out  # type: ignore


# @SeriesWrapper.decorate_accessor("mloc")
class _LocIndexer_unstack_mloc(Generic[T_SeriesWrapper, T_Element]):
    """indexer with pandas index"""

    def __init__(
        self,
        parent: T_SeriesWrapper,
        level: Hashable | Sequence[Hashable] = ("phase",),
    ) -> None:
        self._parent = parent
        self._level = level
        self._index = self._parent.index

        self._index_names = set(self._index.names)
        self._loc = self._parent._series.iloc

    def _get_loc_idx(self, idx: pd.MultiIndex | pd.Index[Any]) -> Any:
        index = self._index
        if isinstance(idx, pd.MultiIndex):
            # names in idx and
            drop: list[Hashable] = list(self._index_names - set(idx.names))
            index = index.droplevel(drop)
            # reorder idx
            idx = idx.reorder_levels(index.names)  # type: ignore
        else:
            drop = list(set(index.names) - {idx.name})
            index = index.droplevel(drop)
        indexer = index.get_indexer_for(idx)  # type: ignore
        return indexer

    def __getitem__(self, idx: pd.MultiIndex | pd.Index[Any]) -> T_SeriesWrapper:
        indexer = self._get_loc_idx(idx)
        out = self._loc[indexer]

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        else:
            raise ValueError("unknown indexer for mloc")
        return out  # type: ignore


class lnPiCollection(SeriesWrapper[lnPiMasked]):
    # class lnPiCollection:
    r"""
    Wrapper around :class:`pandas.Series` for collection of :class:`~lnpy.lnpidata.lnPiMasked` objects.


    Parameters
    ----------
    data : sequence of lnPiMasked
        :math:`\ln \Pi(N)` instances to consider.
    index : array-like, pandas.Index, pandas.MultiIndex, optional
        Index to apply to Series.
    xarray_output : bool, default=True
        If True, then wrap lnPiCollection outputs in :class:`~xarray.DataArray`
    concat_dim : str, optional
        Name of dimensions to concat results along.
        Also Used by :class:`~lnpy.ensembles.xGrandCanonical`.
    concat_coords : string, optional
        parameters `coords `to :func:`xarray.concat`
    unstack : bool, default=True
        If True, then outputs will be unstacked using :meth:`xarray.DataArray.unstack`
    single_state : bool, default=True
        If True, verify that all data has same shape, and value of `state_kws`.
        That is, all ``lnpi`` are for a single state.
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
        data: Sequence[lnPiMasked] | pd.Series[Any],
        index: ArrayLike | pd.Index[Any] | pd.MultiIndex | None = None,
        xarray_output: bool = True,
        concat_dim: str | None = None,
        concat_coords: str | None = None,
        unstack: bool = True,
        **kwargs: Any,
    ) -> None:
        if concat_dim is not None:
            self._concat_dim = concat_dim
        if concat_coords is not None:
            self._concat_coords = concat_coords
        if xarray_output is not None:
            self._xarray_output = xarray_output
        if unstack is not None:
            self._xarray_unstack = unstack

        super().__init__(data=data, index=index, **kwargs)

        # update index name:
        # self._series.index.name = self._concat_dim

    def new_like(
        self,
        data: Sequence[lnPiMasked] | pd.Series[Any] | None = None,
        index: ArrayLike | pd.Index[Any] | pd.MultiIndex | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create new object with optional new data/index."""

        return super().new_like(
            data=data,
            index=index,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords,
            xarray_output=self._xarray_output,
            unstack=self._xarray_unstack,
            **kwargs,
        )

    def _verify_series(self, series: pd.Series[Any]) -> None:
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
    @cached.prop
    def _lnz_series(self) -> pd.Series[Any]:
        return self._series.apply(lambda x: x.lnz)  # type: ignore

    def __repr__(self) -> str:
        return f"<class {self.__class__.__name__}>\n{repr(self._lnz_series)}"

    def __str__(self) -> str:
        return str(self._lnz_series)

    @property
    def state_kws(self) -> dict[str, Any]:
        """state_kws from first :class:`~lnpy.lnpidata.lnPiMasked`"""

        return self.iloc[0].state_kws

    @property
    def nlnz(self) -> int:
        """Number of unique lnzs"""
        return len(self.index.droplevel("phase").drop_duplicates())

    @cached.prop
    def index_frame(self) -> pd.DataFrame:
        """
        Values (from :class:`xarray.DataArray`) for each sample.

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

    @overload
    def _get_lnz(
        self, component: int, *, iloc: int = ..., zloc: int | None = ...
    ) -> float:
        ...

    @overload
    def _get_lnz(
        self, component: None = ..., *, iloc: int = ..., zloc: int | None = ...
    ) -> MyNDArray:
        ...

    def _get_lnz(
        self, component: int | None = None, *, iloc: int = 0, zloc: int | None = None
    ) -> float | MyNDArray:
        """
        Helper function to.
        returns self.iloc[idx].lnz[component]
        """
        if zloc is not None:
            s = self.zloc[zloc]._series
        else:
            s = self._series
        lnz = s.iloc[iloc].lnz
        if component is not None:
            lnz = lnz[component]
        return lnz  # type: ignore

    def _get_level(self, level: str = "phase") -> pd.Index[Any]:
        """Return level values from index"""
        index = self.index
        if isinstance(index, pd.MultiIndex):
            level_idx = index.names.index(level)
            index = index.levels[level_idx]
        return index

    def get_index_level(self, level: str = "phase") -> pd.Index[Any]:
        """Get index values for specified level"""
        return self._get_level(level=level)

    # @cached.prop
    @property
    def _nrec(self) -> int:
        return len(self._series)

    def _lnpi_tot(self, fill_value: float | None = None) -> MyNDArray:
        # old method
        # return np.stack([x.filled() for x in self])

        # new method
        # this is no faster than the original
        # but makes clear where the time is being spent
        first: lnPiMasked = self.iloc[0]
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
    def _lnz_tot(self) -> MyNDArray:
        return np.stack([x.lnz for x in self])

    @property
    def lnz(self) -> MyNDArray:
        return self._lnz_tot

    def _pi_params(
        self, fill_value: float | None = None
    ) -> tuple[MyNDArray, MyNDArray, MyNDArray]:
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

    def wrap_list_results(
        self, items: Sequence[Any] | Sequence[xr.DataArray]
    ) -> Sequence[Any] | xr.DataArray:
        """Utility to wrap output in :class:xarray.DataArray"""
        if self._xarray_output and isinstance(items[0], xr.DataArray):
            return xr.concat(items, self.index, coords=self._concat_coords)  # type: ignore
        else:
            return items

    ##################################################
    # Constructors
    @classmethod
    def from_list(
        cls,
        items: Sequence[lnPiMasked],
        index: Iterable[int] | MyNDArray,
        **kwargs: Any,
    ) -> Self:
        """
        Create collection from list of :class:`~lnpy.lnpidata.lnPiMasked` objects.

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
        lnPiCollection
        """

        df = pd.DataFrame(
            [lnpi._index_dict(phase) for lnpi, phase in zip(items, index)]
        )
        new_index = pd.MultiIndex.from_frame(df)
        return cls(data=items, index=new_index, **kwargs)

    @classmethod
    def from_builder(
        cls,
        lnzs: Sequence[float] | MyNDArray,
        # TODO: make better type for build_phases.
        build_phases: Callable[..., tuple[list[lnPiMasked], MyNDArray]],
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
        nmax: int | None = None,
        # concat_kws: Mapping[str, Any] |None=None,
        base_class: str | type = "first",
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """
        Build collection from scalar builder

        Parameters
        ----------
        lnzs : sequence of float
            One dimensional array of lnz value for the varying component.
        ref : lnPiMasked
            lnpi_phases to reweight to get list of lnpi's
        build_phases : callable
            Typically one of `PhaseCreator.build_phases_mu` or `PhaseCreator.build_phases_dmu`
        build_kws : optional
            optional arguments to `build_phases`

        Returns
        -------
        lnPiCollection


        See Also
        --------
        ~lnpy.segment.PhaseCreator.build_phases
        ~lnpy.segment.PhaseCreator.build_phases_mu
        ~lnpy.segment.PhaseCreator.build_phases_dmu

        """
        if build_kws is None:
            build_kws = {}

        build_kws = dict(build_kws, phases_factory=False)
        seq = get_tqdm(lnzs, desc="build")
        L = parallel_map(build_phases, seq, ref=ref, nmax=nmax, **build_kws)
        # return cls.concat(L, verify=verify, concat_kws=concat_kws, base_class=base_class,
        #                   *args, **kwargs)

        items: list[lnPiMasked] = []
        index: list[int] = []
        for data, idx in L:
            items += data
            index += list(idx)
        return cls.from_list(items, index, base_class=base_class, *args, **kwargs)

    ################################################################################
    # dataarray io
    def to_dataarray(
        self, dtype: DTypeLike | None = None, reset_index: bool = True
    ) -> xr.DataArray:
        """
        Convert collection to a :class:`~xarray.DataArray`

        Parameters
        ----------
        dtype : `numpy.dtype`, optional
            Default to `numpy.uint8`.
        reset_index : bool, default=True
        """

        dtype = dtype or np.uint8

        labels = []
        indexes = []

        for _, g in self.groupby_allbut("phase"):
            indexes.append(g.index[[0]])

            features = np.array(g.index.get_level_values("phase")) + 1
            masks = [x.mask for x in g]
            labels.append(
                masks_to_labels(masks, features=features, convention=False, dtype=dtype)
            )

        index = indexes[0].append(indexes[1:])  # type: ignore

        data = np.stack(labels)

        out = (
            xr.DataArray(
                data,
                dims=self.xge.dims_rec + self.xge.dims_n,
                name="labels",
            )
            .assign_coords(**{self._concat_dim: index, **self.state_kws})
            .assign_attrs(**self.xge._standard_attrs)
        )

        if reset_index:
            out = out.reset_index(self._concat_dim)

        return out

    @classmethod
    @docfiller.decorate
    def from_labels(
        cls,
        ref: lnPiMasked,
        labels: Sequence[MyNDArray],
        lnzs: Sequence[float | MyNDArray] | MyNDArray,
        features: Sequence[int] | MyNDArray | None = None,
        include_boundary: bool = False,
        labels_kws: Mapping[str, Any] | None = None,
        check_features: bool = True,
        **kwargs: Any,
    ) -> Self:
        r"""
        Create from reference :class:`~lnpy.lnpidata.lnPiMasked` and labels array


        Parameters
        ----------
        ref : lnPiMasked
        labels : sequence of ndarray of int
            Each ``labels[i]`` is a labels array for each value of ``lnzs[i]``.
            That is, the labels for different phases at a given value of `lnz`.
        lnzs : sequence
            Each lnzs[i] will be passed to ``ref.reweight``.
        {features}
        {include_boundary}
        labels_kws : mapping, optional
        {check_features}
        **kwargs
            Extra arguments past to :meth:`from_list`

        See Also
        --------
        ~lnpy.utils.labels_to_masks
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
                **labels_kws,
            )

            index = list(np.array(features_tmp) - 1)
            items += lnpi.list_from_masks(masks, convention=False)
            indexes += index

        return cls.from_list(items=items, index=indexes, **kwargs)

    @classmethod
    @docfiller.decorate
    def from_dataarray(
        cls,
        ref: lnPiMasked,
        da: xr.DataArray,
        grouper: Hashable = "sample",
        include_boundary: bool = False,
        labels_kws: Mapping[str, Any] | None = None,
        features: Sequence[int] | MyNDArray | None = None,
        check_features: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Create a collection from DataArray of labels

        Parameters
        ----------
        ref : lnPiMasked
        da : DataArray or int
            Labels.
        grouper : Hashable
            Name of dimension(s) to group along to give a single label array
        {features}
        {check_features}


        See Also
        --------
        from_labels


        """

        labels = []
        lnzs = []

        for _, g in da.groupby(grouper):
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
            **kwargs,
        )  # yapf: disable

    @cached.prop
    @docfiller.decorate
    def xge(self) -> ensembles.xGrandCanonical:
        """{accessor.xge}"""
        from .ensembles import xGrandCanonical

        return xGrandCanonical(self)

    @cached.prop
    def wfe(self) -> lnpienergy.wFreeEnergyCollection:
        """Accessor to :class:`~lnpy.lnpienergy.wFreeEnergyPhases` from :attr:`wfe_phases`."""
        from .lnpienergy import wFreeEnergyCollection

        return wFreeEnergyCollection(self)

    @cached.prop
    def wfe_phases(self) -> lnpienergy.wFreeEnergyPhases:
        """Accessor to :class:`~lnpy.lnpienergy.wFreeEnergyCollection` from :attr:`wfe`."""
        from .lnpienergy import wFreeEnergyPhases

        return wFreeEnergyPhases(self)

    @property
    def wlnPi(self) -> lnpienergy.wFreeEnergyCollection:
        """
        Deprecated accessor to :class:`~lnpy.lnpienergy.wFreeEnergyCollection` from :attr:`wlnPi`.

        Alias to :attr:`wfe`
        """
        warn("Using `wlnPi` accessor is deprecated.  Please use `wfe` accessor instead")
        return self.wfe

    @property
    def wlnPi_single(self) -> lnpienergy.wFreeEnergyPhases:
        """
        Deprecated accessor to :class:`~lnpy.lnpienergy.wFreeEnergyPhases` from :attr:`wlnPi_single`.

        Alias to :attr:`wfe_phases`
        """
        warn("Using `wlnPi_single is deprecated.  Please use `self.wfe_phases` instead")
        return self.wfe_phases

    @cached.prop
    def spinodal(self) -> stability.Spinodals:
        """Accessor to :class:`~lnpy.stability.Spinodals`"""
        from .stability import Spinodals

        return Spinodals(self)

    @cached.prop
    def binodal(self) -> stability.Binodals:
        """Accessor to :class:`~lnpy.stability.Binodals`"""
        from .stability import Binodals

        return Binodals(self)

    def stability_append(
        self,
        other: Self | None,
        append: bool = True,
        sort: bool = True,
        copy_stability: bool = True,
    ) -> Self:
        if (not append) and (not copy_stability):
            raise ValueError("one of append or copy_stability must be True")

        if other is None:
            other = self
        spin = other.spinodal
        bino = other.binodal
        if append:
            new = self.append(spin.appender).append(bino.appender)  # type: ignore
            if sort:
                new = new.sort_index()
        else:
            new = self.copy()
        if copy_stability:
            # TODO: fix this hack
            new._cache["spinodal"] = spin
            new._cache["binodal"] = bino
            # new.spinodal = spin
            # new.binodal = bino
        return new


################################################################################
# Accessors for ColleectionlnPi
# lnPiCollection.register_accessor("xge", xge_accessor)
# lnPiCollection.register_accessor("wfe", wfe_accessor)
# lnPiCollection.register_accessor("wfe_phases", wfe_phases_accessor)
# lnPiCollection.register_accessor("wlnPi", wlnPi_accessor)
# lnPiCollection.register_accessor("wlnPi_single", wlnPi_single_accessor)
