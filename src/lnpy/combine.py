# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
r"""
Routines to combine :math:`\ln \Pi` data (:mod:`~lnpy.combine`)
===============================================================

"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterator, TypeVar, cast, overload

import numpy as np
import pandas as pd
import xarray as xr
from module_utilities.docfiller import DocFiller
from scipy.sparse import coo_array

from .docstrings import docfiller
from .utils import peek_at

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Sequence

    from numpy.typing import NDArray

    T_Array = TypeVar("T_Array", pd.Series[Any], NDArray[Any], xr.DataArray)
    T_Series = TypeVar("T_Series", pd.Series[Any], xr.DataArray)
    T_Frame = TypeVar("T_Frame", pd.DataFrame, xr.Dataset)

    T_Dataset_Array = TypeVar("T_Dataset_Array", xr.DataArray, xr.Dataset)

    T_Frame_Array = TypeVar("T_Frame_Array", pd.DataFrame, xr.DataArray, xr.Dataset)


_docstrings_local = r"""
Parameters
----------
lnpi_name :
    Column name corresponding to :math:`\ln \Pi`.
window_name :
    Column name corresponding to "window", i.e., an individual simulation.
    Note that this is only used if passing in a single dataframe with multiple windows.
state_name :
    Column name corresponding to simulation state. For example, ``state="state"``.
macrostate_names :
    Column name(s) corresponding to a single "state". For example, for a single
    component system, this could be ``macrostate_names="n"``, and for a binary
    system ``macrostate_names=["n_0", "n_1"]``
up_name :
    Column name corresponding to "up" probability.
down_name :
    Column name corresponding to "down" probability.
weight_name :
    Column name corresponding to "weight" of probability.
table_assign | table :
    :class:`pandas.DataFrame` or :class:`xarray.Dataset` data container.
check_connected :
    If ``True``, check that all windows form a connected graph.
tables :
    Individual sample windows. If pass in a single
    :class:`~pandas.DataFrame`, it must contain the column ``window_name``.
    Otherwise, the individual frames will be concatenated and the
    ``window_name`` column will be added (or replaced if already present).
up :
    Probability of moving from ``state[i]`` to ``state[i+1]``.
down :
    Probability of moving from ``state[i]`` to ``state[i-1]``.
norm :
    If True, normalize :math:`\ln \Pi(N)`.
array_name | name:
    Optional name to assign to the output :class:`pandas.Series` or `xarray.DataArray`.


Raises
------
OverlapError
    If the overlaps do not form a connected graph, then raise a ``OverlapError``.

"""


docfiller_local = docfiller.append(
    DocFiller.from_docstring(_docstrings_local, combine_keys="parameters")
).decorate


class OverlapError(ValueError):
    """Specific error for missing overlaps."""


def _str_or_iterable(x: str | Iterable[str]) -> list[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


# * connected graph
# From networkx (see https://github.com/networkx/networkx/blob/main/networkx/algorithms/components/connected.py)
def _plain_bfs(adj: dict[int, set[int]], source: int) -> set[int]:
    """A fast BFS node generator"""
    n = len(adj)
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen


def _connected_components(adj: dict[int, set[int]]) -> Iterator[set[int]]:
    seen = set()
    for v in adj:
        if v not in seen:
            c = _plain_bfs(adj, v)
            seen.update(c)
            yield c


def _build_graph(nodes: Iterable[int], edges: NDArray[Any]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {node: set() for node in nodes}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def check_windows_overlap(
    overlap_table: pd.DataFrame,
    windows: Iterable[int],
    window_index_name: str,
    macrostate_names: str | Iterable[str] = "state",
    verbose: bool = True,
) -> None:
    """
    Check that window overlaps form a connected graph.

    Parameters
    ----------
    overlap_table: DataFrame
        Frame should contain columns ``macrostate_names`` and ``window_index_name``
        and rows corresponding to overlaps.

    Raises
    ------
    OverlapError
        If the overlaps do not form a connected graph, then raise a ``OverlapError``.
    """
    macrostate_names = _str_or_iterable(macrostate_names)
    overlap_table = overlap_table[[window_index_name, *macrostate_names]]

    x: pd.DataFrame = (
        overlap_table.merge(
            overlap_table, on=macrostate_names, how="outer", suffixes=("", "_nebr")
        )
        .drop(macrostate_names, axis=1)
        .query(f"{window_index_name} < {window_index_name}_nebr")
        .drop_duplicates()
    )

    graph = _build_graph(nodes=windows, edges=x.to_numpy())

    components = list(_connected_components(graph))

    if len(components) != 1:
        msg = "Disconnected graph."
        if verbose:
            for subgraph in components:
                msg = f"{msg}\ngraph: {subgraph}"
        raise OverlapError(msg)


# * concat objects
def _concat_windows_dataframe(
    first: pd.DataFrame,
    tables: Iterator[pd.DataFrame],
    window_name: str,
    overwrite_window: bool = True,
) -> pd.DataFrame:
    """
    Concat windows of dataframes.

    Should use first, tables = peek_at(tables).
    """
    table = pd.concat(
        dict(enumerate(tables)),
        names=[window_name, *first.index.names],
    )
    if window_name in table.columns:
        if overwrite_window:
            return table.drop(window_name, axis=1).reset_index(window_name)
        return table.reset_index(window_name, drop=True)

    return table.reset_index(window_name)


def _concat_windows_xarray(
    first: T_Dataset_Array,
    tables: Iterator[T_Dataset_Array],
    window_name: str,
    coord_names: str | Iterable[str],
    index_name: str,
    overwrite_window: bool = True,
) -> T_Dataset_Array:
    if index_name not in first.coords:
        # stack each object
        # if wait until end, dtype might change because of missing values during concat and stack...
        def _process_object(obj: T_Dataset_Array, window: int) -> T_Dataset_Array:
            if window_name not in obj.dims:
                return obj.expand_dims(window_name).assign_coords(
                    {window_name: (window_name, [window])}
                )
            if overwrite_window:
                return obj.assign_coords(
                    {window_name: xr.full_like(obj[window_name], fill_value=window)}  # pyright: ignore[reportIndexIssue]
                )
            return obj

        out = xr.concat(  # pyright: ignore[reportCallIssue]
            (
                (
                    _process_object(ds, window).stack(  # noqa: PD013
                        {index_name: [window_name, *_str_or_iterable(coord_names)]}
                    )  # pyright: ignore[reportArgumentType]
                )
                for window, ds in enumerate(tables)
            ),
            dim=index_name,
        )
    else:
        out = xr.concat(tables, dim=index_name)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return out  # type: ignore[no-any-return]


@overload
def concat_windows(
    tables: Iterable[xr.Dataset],
    window_name: str = ...,
    coord_names: str | Iterable[str] = ...,
    index_name: str = ...,
    overwrite_window: bool = ...,
) -> xr.Dataset: ...


@overload
def concat_windows(
    tables: Iterable[xr.DataArray],
    window_name: str = ...,
    coord_names: str | Iterable[str] = ...,
    index_name: str = ...,
    overwrite_window: bool = ...,
) -> xr.DataArray: ...


@overload
def concat_windows(
    tables: Iterable[pd.DataFrame],
    window_name: str = ...,
    coord_names: str | Iterable[str] = ...,
    index_name: str = ...,
    overwrite_window: bool = ...,
) -> pd.DataFrame: ...


def concat_windows(
    tables: Iterable[xr.Dataset] | Iterable[xr.DataArray] | Iterable[pd.DataFrame],
    window_name: str = "window",
    coord_names: str | Iterable[str] = "state",
    index_name: str = "index",
    overwrite_window: bool = True,
) -> xr.Dataset | xr.DataArray | pd.DataFrame:
    """
    Concatenate object windows to single stacked object.

    Parameters
    ----------
    tables :
        Each object is for a single window.
    window_name :
        Name of column or coordinate to track window. This will the index of
        the window in ``tables``.
    coord_names :
        Name of additional coordinates for :mod:`xarray` objects.
    index_name :
        Name of stacked multiindex coordinate for :mod:`xarray` objects.

    Returns
    -------
    Dataset or DataArray or DataFrame
        Concatenated object with ``window_name`` added. For :mod:`xarray`
        output, stack ``window_name`` and ``coord_names`` to ``index_name``.

    Examples
    --------
    >>> table = pd.DataFrame({"state": range(5), "rec": [0] * 5, "values": range(5)})
    >>> tables = [table.iloc[:3], table.iloc[2:]]
    >>> tables[0]
       state  rec  values
    0      0    0       0
    1      1    0       1
    2      2    0       2
    >>> tables[1]
       state  rec  values
    2      2    0       2
    3      3    0       3
    4      4    0       4
    >>> concat_windows(tables)
       window  state  rec  values
    0       0      0    0       0
    1       0      1    0       1
    2       0      2    0       2
    2       1      2    0       2
    3       1      3    0       3
    4       1      4    0       4

    To keep the supplied window value, use ``overwrite_window=False``
    >>> tables_with_window = [
    ...     x.assign(window=name) for x, name in zip(tables, ["a", "b"])
    ... ]
    >>> print(tables_with_window[0])
       state  rec  values window
    0      0    0       0      a
    1      1    0       1      a
    2      2    0       2      a
    >>> print(tables_with_window[1])
       state  rec  values window
    2      2    0       2      b
    3      3    0       3      b
    4      4    0       4      b
    >>> # Identical to above without the option
    >>> concat_windows(tables_with_window)
       window  state  rec  values
    0       0      0    0       0
    1       0      1    0       1
    2       0      2    0       2
    2       1      2    0       2
    3       1      3    0       3
    4       1      4    0       4
    >>> concat_windows(tables_with_window, overwrite_window=False)
       state  rec  values window
    0      0    0       0      a
    1      1    0       1      a
    2      2    0       2      a
    2      2    0       2      b
    3      3    0       3      b
    4      4    0       4      b

    This also works for :class:`~xarray.Dataset` and :class:`~xarray.DataArray`
    objects. In this case, ``window_name`` will be added to ``coord_names`` and
    put in the multiindex ``index_name``.
    >>> datasets = [x.set_index("state").to_xarray() for x in tables]
    >>> datasets[0]
    <xarray.Dataset> Size: 72B
    Dimensions:  (state: 3)
    Coordinates:
      * state    (state) int64 24B 0 1 2
    Data variables:
        rec      (state) int64 24B 0 0 0
        values   (state) int64 24B 0 1 2
    >>> datasets[1]
    <xarray.Dataset> Size: 72B
    Dimensions:  (state: 3)
    Coordinates:
      * state    (state) int64 24B 2 3 4
    Data variables:
        rec      (state) int64 24B 0 0 0
        values   (state) int64 24B 2 3 4

    >>> concat_windows(datasets, coord_names="state")
    <xarray.Dataset> Size: 240B
    Dimensions:  (index: 6)
    Coordinates:
      * index    (index) object 48B MultiIndex
      * window   (index) int64 48B 0 0 0 1 1 1
      * state    (index) int64 48B 0 1 2 2 3 4
    Data variables:
        rec      (index) int64 48B 0 0 0 0 0 0
        values   (index) int64 48B 0 1 2 2 3 4


    This also works for multiple coordinates and assigned windows
    >>> data = [
    ...     x.assign(window=name)
    ...     .set_index(["rec", "window", "state"])
    ...     .to_xarray()["values"]
    ...     for x, name in zip(tables, ["a", "b"])
    ... ]
    >>> data[0]
    <xarray.DataArray 'values' (rec: 1, window: 1, state: 3)> Size: 24B
    array([[[0, 1, 2]]])
    Coordinates:
      * rec      (rec) int64 8B 0
      * window   (window) object 8B 'a'
      * state    (state) int64 24B 0 1 2
    >>> data[1]
    <xarray.DataArray 'values' (rec: 1, window: 1, state: 3)> Size: 24B
    array([[[2, 3, 4]]])
    Coordinates:
      * rec      (rec) int64 8B 0
      * window   (window) object 8B 'b'
      * state    (state) int64 24B 2 3 4
    >>> concat_windows(data, coord_names=["rec", "state"], overwrite_window=False)
    <xarray.DataArray 'values' (index: 6)> Size: 48B
    array([0, 1, 2, 2, 3, 4])
    Coordinates:
      * index    (index) object 48B MultiIndex
      * window   (index) object 48B 'a' 'a' 'a' 'b' 'b' 'b'
      * rec      (index) int64 48B 0 0 0 0 0 0
      * state    (index) int64 48B 0 1 2 2 3 4


    Things work as expected if data is already stacked. This simply calls
    :func:`xarray.concat` with ``dim=index_name``.
    >>> data = [
    ...     x.assign(window=name)
    ...     .to_xarray()
    ...     .set_index(index=["rec", "window", "state"])
    ...     for x, name in zip(tables, ["a", "b"])
    ... ]
    >>> data[0]
    <xarray.Dataset> Size: 120B
    Dimensions:  (index: 3)
    Coordinates:
      * index    (index) object 24B MultiIndex
      * rec      (index) int64 24B 0 0 0
      * window   (index) object 24B 'a' 'a' 'a'
      * state    (index) int64 24B 0 1 2
    Data variables:
        values   (index) int64 24B 0 1 2
    >>> data[1]
    <xarray.Dataset> Size: 120B
    Dimensions:  (index: 3)
    Coordinates:
      * index    (index) object 24B MultiIndex
      * rec      (index) int64 24B 0 0 0
      * window   (index) object 24B 'b' 'b' 'b'
      * state    (index) int64 24B 2 3 4
    Data variables:
        values   (index) int64 24B 2 3 4
    >>> concat_windows(data, index_name="index")
    <xarray.Dataset> Size: 240B
    Dimensions:  (index: 6)
    Coordinates:
      * index    (index) object 48B MultiIndex
      * rec      (index) int64 48B 0 0 0 0 0 0
      * window   (index) object 48B 'a' 'a' 'a' 'b' 'b' 'b'
      * state    (index) int64 48B 0 1 2 2 3 4
    Data variables:
        values   (index) int64 48B 0 1 2 2 3 4

    """

    first, tables_iter = peek_at(tables)

    if isinstance(first, xr.Dataset):
        return _concat_windows_xarray(
            first=first,
            tables=cast("Iterator[xr.Dataset]", tables_iter),
            index_name=index_name,
            window_name=window_name,
            coord_names=coord_names,
            overwrite_window=overwrite_window,
        )

    if isinstance(first, xr.DataArray):
        return _concat_windows_xarray(
            first=first,
            tables=cast("Iterator[xr.DataArray]", tables_iter),
            index_name=index_name,
            window_name=window_name,
            coord_names=coord_names,
            overwrite_window=overwrite_window,
        )

    if isinstance(first, pd.DataFrame):  # progma: no branch
        return _concat_windows_dataframe(
            first=first,
            tables=cast("Iterator[pd.DataFrame]", tables_iter),
            window_name=window_name,
            overwrite_window=overwrite_window,
        )

    msg = f"Unknown element type {type(first)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


# * Shift lnPi
# ** Matrix Equation construction
def _add_window_index(
    table: pd.DataFrame,
    window_name: str,
    window_index_name: str,
) -> pd.DataFrame:
    """
    Add continuous integer index to the provided windows.

    This will fix cases where windows = ['a', 'b', ...], etc.
    """
    window_map: pd.Series[Any] = (
        table[window_name]
        .drop_duplicates()
        .pipe(lambda x: pd.Series(range(len(x)), index=x))
    )
    table[window_index_name] = window_map[table[window_name]].to_numpy()
    return table


def _create_overlap_table(
    table: pd.DataFrame,
    window_index_name: str,
    window_max: int,
    macrostate_names: list[str],
    lnpi_name: str,
    check_connected: bool = True,
) -> pd.DataFrame:
    overlap_table = table.loc[
        table[macrostate_names].duplicated(keep=False),
        [window_index_name, *macrostate_names, lnpi_name],
    ]

    if len(overlap_table) == 0:
        msg = "No overlaps with multiple windows"
        raise OverlapError(msg)

    if check_connected:
        check_windows_overlap(
            windows=range(window_max + 1),
            overlap_table=overlap_table,
            window_index_name=window_index_name,
            macrostate_names=macrostate_names,
        )
    return overlap_table


def _create_overlap_total_table(
    overlap_table: pd.DataFrame,
    window_index_name: str,
    window_max: int,
    macrostate_names: list[str],
    lnpi_name: str,
) -> pd.DataFrame:
    group = overlap_table.groupby(macrostate_names, as_index=False)[lnpi_name]
    return (
        overlap_table.assign(
            total=group.transform("sum"),
            count=group.transform("count"),
        )
        # limit to only window < window_max.
        # Equivalent to not shifting last window.
        .query(f"{window_index_name} < {window_max}")
        # assign eq_idx
        .assign(eq_idx=lambda x: range(len(x)))
    )


def _create_overlap_outer_table(
    overlap_total_table: pd.DataFrame,
    macrostate_names: list[str],
    window_index_name: str,
) -> pd.DataFrame:
    return overlap_total_table[[*macrostate_names, "eq_idx"]].merge(
        overlap_total_table[[*macrostate_names, window_index_name]],
        on=macrostate_names,
        how="outer",
    )


def _create_lhs_matrix_sparse(
    overlap_total_table: pd.DataFrame,
    overlap_outer_table: pd.DataFrame,
    window_index_name: str,
    window_max: int,
) -> coo_array:
    coeff_table = pd.concat(
        (
            overlap_total_table[["eq_idx", window_index_name, "count"]],
            overlap_outer_table[["eq_idx", window_index_name]].assign(count=-1),
        )
    )

    return coo_array(
        (
            coeff_table["count"],
            (coeff_table["eq_idx"], coeff_table[window_index_name]),
        ),
        shape=(len(overlap_total_table), window_max),
    )


def _create_lhs_matrix_numpy(
    overlap_total_table: pd.DataFrame,
    overlap_outer_table: pd.DataFrame,
    window_index_name: str,
    window_max: int,
) -> NDArray[Any]:
    overlap_outer_table = (
        overlap_outer_table[["eq_idx", window_index_name]]
        .assign(count=1)
        .groupby(["eq_idx", window_index_name], as_index=False)
        .sum()
    )
    a = np.zeros((len(overlap_total_table), window_max))
    a[overlap_total_table["eq_idx"], overlap_total_table[window_index_name]] = (
        overlap_total_table["count"]
    )
    a[overlap_outer_table["eq_idx"], overlap_outer_table[window_index_name]] -= (
        overlap_outer_table["count"]
    )
    return a


@docfiller_local
def shift_lnpi_windows(
    table: pd.DataFrame,
    macrostate_names: str | Iterable[str] = "state",
    lnpi_name: str = "ln_prob",
    window_name: str = "window",
    use_sparse: bool = True,
    check_connected: bool = False,
) -> pd.DataFrame:
    r"""
    Shift :math:`\ln \Pi` windows by minimizing error between average and each window.

    This performs least squares on the problem:

    .. math::

        \min_{{C_0, C_1, ..., C_{{W-1}}}} \sum_{{\rm{{overlap}}_m \in \rm{{overlaps}}}}
        \, \sum_{{N_m, k \in \rm{{overlap}}_m}} [\ln \bar{{\Pi}}(N_m) - (\ln \Pi_k
        (N_m) + C_k)]^2

    where,

    - :math:`C_j` : shift for sample :math:`j`. :math:`W` : number of
    - overlapping samples :math:`\Pi_k(N)` : transition matrix at particle
    - number :math:`N` for the kth sample :math:`\rm{{overlap}}_m` : a particular
    - overlap at particle number :math:`N_m` and over samples :math:`k`,
    - :math:`\ln \bar{{\Pi}}` is the to be determined average value.

    This can be reduced to a matrix problem of the form:

    .. math::

        S C_j - \sum_{{k \in \rm{{overlap}}_m}} C_k = - (S \ln \Pi_j(N_m) - \sum_{{k
        \in \rm{{overlap}}_m}} \ln \Pi_k(N_m))

    the sum runs over all samples with overlap at state :math:`N_m`, :math:`S`
    is the number of such overlaps (i.e., :math:`S = \sum_{{k \in
    \rm{{overlap}}_m}} 1`). There are such equations for all :math:`j \in
    \rm{{overlap}}_m`.


    Parameters
    ----------
    {tables}
    {macrostate_names}
    {lnpi_name}
    {window_name}
    use_sparse : bool, default=True
        Use :class:`~scipy.sparse.coo_array` array in matrix equation. This is
        often faster than using a :class:`numpy.ndarray`.
    {check_connected}

    Returns
    -------
    DataFrame
        Combined table with appropriately shifted ``lnpi_name`` column.
        Note that the table is not yet averaged over ``macrostate_names``.

    See Also
    --------
    concat_window
    keep_first

    Examples
    --------
    >>> states = pd.DataFrame(range(5), columns=["state"])
    >>> table = concat_windows(
    ...     [
    ...         table.assign(lnpi=lambda x: x["state"] + i * 10)
    ...         for i, table in enumerate([states.iloc[:3], states.iloc[2:]])
    ...     ]
    ... )
    >>> table
       window  state  lnpi
    0       0      0     0
    1       0      1     1
    2       0      2     2
    2       1      2    12
    3       1      3    13
    4       1      4    14
    >>> shifted = shift_lnpi_windows(table, lnpi_name="lnpi")
    >>> shifted
       window  state  lnpi
    0       0      0   0.0
    1       0      1   1.0
    2       0      2   2.0
    2       1      2   2.0
    3       1      3   3.0
    4       1      4   4.0

    Note that the resulting dataframe includes all (properly shifted) data.
    To, for example, average over separate windows, use something like:
    >>> shifted.groupby("state", as_index=False).mean()
       state  window  lnpi
    0      0     0.0   0.0
    1      1     0.0   1.0
    2      2     0.5   2.0
    3      3     1.0   3.0
    4      4     1.0   4.0
    """
    window_index_name = "_window_index"

    if window_name not in table.columns:
        msg = f"Passed single table must contain {window_name=} column."
        raise ValueError(msg)
    table = _add_window_index(
        table, window_name=window_name, window_index_name=window_index_name
    )

    window_max = cast(int, table[window_index_name].iloc[-1])
    if window_max == 0:
        return table

    macrostate_names = _str_or_iterable(macrostate_names)
    overlap_total_table = _create_overlap_total_table(
        overlap_table=_create_overlap_table(
            table,
            window_index_name=window_index_name,
            window_max=window_max,
            macrostate_names=macrostate_names,
            lnpi_name=lnpi_name,
            check_connected=check_connected,
        ),
        window_index_name=window_index_name,
        window_max=window_max,
        macrostate_names=macrostate_names,
        lnpi_name=lnpi_name,
    )

    b: NDArray[Any] = overlap_total_table.pipe(
        lambda x: -x["count"] * x[lnpi_name] + x["total"]
    ).to_numpy()

    overlap_outer_table = _create_overlap_outer_table(
        overlap_total_table=overlap_total_table,
        macrostate_names=macrostate_names,
        window_index_name=window_index_name,
    )

    if use_sparse:
        a = _create_lhs_matrix_sparse(
            overlap_total_table=overlap_total_table,
            overlap_outer_table=overlap_outer_table,
            window_index_name=window_index_name,
            window_max=window_max,
        )

        lhs = (a.T @ a).toarray()
        # There's a bug with multiplying a shape=(1,1) a into b.
        # The result will be a scalar.
        # so make sure its a vector
        rhs = np.atleast_1d(a.T @ b)

    else:
        a = _create_lhs_matrix_numpy(
            overlap_total_table=overlap_total_table,
            overlap_outer_table=overlap_outer_table,
            window_index_name=window_index_name,
            window_max=window_max,
        )

        lhs = a.T @ a
        rhs = a.T @ b

    shift = np.zeros(window_max + 1)
    shift[:-1] = np.linalg.solve(lhs, rhs)
    shift -= shift[0]

    return table.assign(
        **{lnpi_name: table[lnpi_name] + shift[table[window_index_name].to_numpy()]}
    ).drop(window_index_name, axis=1)


# * keep_first
def _filter_min_max_keep_first(
    table: pd.DataFrame,
    window_name: str,
    state_name: str,
    window_index_name: str,
    check_connected: bool,
) -> pd.DataFrame:
    window_map = (
        table[[window_name, state_name]]
        .groupby(window_name, as_index=False)
        .min()
        .sort_values(state_name)[window_name]
        .pipe(lambda x: pd.Series(range(len(x)), index=x))
    )

    window_max = window_map.iloc[-1]
    if window_max == 0:
        return table

    table = table.assign(
        **{window_index_name: window_map[table[window_name]].to_numpy()}
    )

    if check_connected:
        _create_overlap_table(
            table,
            window_index_name=window_index_name,
            window_max=window_max,
            macrostate_names=[state_name],
            lnpi_name=window_index_name,
            check_connected=check_connected,
        )

    # remove overlap from beginning of each window
    min_max = table.groupby(window_index_name, as_index=False)[state_name].agg(
        ["min", "max"]
    )
    keep_min = min_max["max"].shift(1, fill_value=-100)
    keep_max = min_max["max"]
    query = f"_keep_min < {state_name} <= _keep_max"

    return (
        table.assign(
            _keep_min=lambda x: keep_min[x[window_index_name]].to_numpy(),
            _keep_max=lambda x: keep_max[x[window_index_name]].to_numpy(),
        )
        .query(query)
        .drop(["_keep_min", "_keep_max", window_index_name], axis=1)
    )


# * Collection matrix
@docfiller_local
def keep_first(
    table: T_Frame_Array,
    window_name: str = "window",
    state_name: str = "state",
    check_connected: bool = False,
    index_name: str = "index",
    reset_window: bool = True,
) -> T_Frame_Array:
    """
    Keep overlaps in the "first" window and drop in subsequent windows.

    For example, if have two windows `A` and `B` with states `state_A=[0,1,2]`
    and `state_B=[1,2,3]` and observable `x_A(state_A)`, `x_B(state_B)`, then
    the combined result will be `state=[0,1,2,3]`, `x = [x_A(0), x_A(1),
    x_A(2), x_B(3)]`.

    Note that the windows are first sorted by the minimum value of ``state_name``.

    Parameters
    ----------
    table :
        Input data. If :class:`~pandas.DataFrame`, must have columns
        ``window_name`` and ``state_name``. If :class:`~xarray.Dataset` or
        :class:`~xarray.DataArray`, must have either stacked index
        ``index_name`` that contains ``window_name`` and ``state_name``, or
        ``window_name`` and ``state_name`` will be stacked to ``index_name``.
    {window_name}
    {state_name}
    {check_connected}
    index_name :
        When passing in :class:`~xarray.Dataset` or :class:`~xarray.DataArray`,
        ``window_name`` and ``state_name`` will be stacked into a single
        multiindex of name ``index_name``. The passed objects can either
        already have the stacked ``index_name`` or it will be created.
    reset_window :
        If ``True`` remove ``window_name`` from ``index_name``. If resulting
        multiindex only contains ``state_name``, then reset this as well. Only
        applies to :mod:`xarray` objects.

    Returns
    -------
    DataFrame or Dataset or DataArray
        Same type as input ``table``

    Note
    ----
    If there is not expanded ensemble sampling (i.e., non-integer ``state``
    values) in windows, you should prefer using :func:`updown_mean`.

    See Also
    --------
    concat_windows
    shift_lnpi_windows

    Examples
    --------
    >>> states = pd.DataFrame(range(5), columns=["state"])
    >>> tables = [
    ...     x.assign(lnpi=lambda x: x["state"] + i * 10)
    ...     for i, x in enumerate([states.iloc[:3], states.iloc[2:]])
    ... ]
    >>> table = concat_windows(tables)
    >>> table
       window  state  lnpi
    0       0      0     0
    1       0      1     1
    2       0      2     2
    2       1      2    12
    3       1      3    13
    4       1      4    14
    >>> keep_first(table)
       window  state  lnpi
    0       0      0     0
    1       0      1     1
    2       0      2     2
    3       1      3    13
    4       1      4    14


    The actual value of ``window_name`` does not matter.
    Windows are sorted by the minimum value of ``state_name``
    for each unique window.

    >>> table = concat_windows(tables[-1::-1])
    >>> table
       window  state  lnpi
    2       0      2    12
    3       0      3    13
    4       0      4    14
    0       1      0     0
    1       1      1     1
    2       1      2     2
    >>> keep_first(table)
       window  state  lnpi
    3       0      3    13
    4       0      4    14
    0       1      0     0
    1       1      1     1
    2       1      2     2

    This also works with :class:`~xarray.Dataset` and
    :class:`~xarray.DataArray` objects. For example

    >>> data = concat_windows([df.set_index("state").to_xarray() for df in tables])
    >>> data
    <xarray.Dataset> Size: 192B
    Dimensions:  (index: 6)
    Coordinates:
      * index    (index) object 48B MultiIndex
      * window   (index) int64 48B 0 0 0 1 1 1
      * state    (index) int64 48B 0 1 2 2 3 4
    Data variables:
        lnpi     (index) int64 48B 0 1 2 12 13 14
    >>> keep_first(data)
    <xarray.Dataset> Size: 120B
    Dimensions:  (state: 5)
    Coordinates:
      * state    (state) int64 40B 0 1 2 3 4
        window   (state) int64 40B 0 0 0 1 1
    Data variables:
        lnpi     (state) int64 40B 0 1 2 13 14
    >>> keep_first(data["lnpi"])
    <xarray.DataArray 'lnpi' (state: 5)> Size: 40B
    array([ 0,  1,  2, 13, 14])
    Coordinates:
      * state    (state) int64 40B 0 1 2 3 4
        window   (state) int64 40B 0 0 0 1 1


    Note that internally, ``state_name`` and ``window_name`` are stacked into a
    multiindex. ``window_name`` is by default removed from the index after
    combining. To keep the multiindex, use:

    >>> keep_first(data, reset_window=False)
    <xarray.Dataset> Size: 160B
    Dimensions:  (index: 5)
    Coordinates:
      * index    (index) object 40B MultiIndex
      * window   (index) int64 40B 0 0 0 1 1
      * state    (index) int64 40B 0 1 2 3 4
    Data variables:
        lnpi     (index) int64 40B 0 1 2 13 14

    """

    window_index_name = "_window_index"

    def _process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        if window_name not in data.columns:
            msg = f"Passed single table must contain {window_name=} column."
            raise ValueError(msg)

        return _filter_min_max_keep_first(
            table=data,
            window_name=window_name,
            state_name=state_name,
            window_index_name=window_index_name,
            check_connected=check_connected,
        )

    def _process_xarray(data: T_Dataset_Array) -> T_Dataset_Array:
        # make sure correct
        if index_name in data.coords:
            names = data.indexes[index_name].names
            if window_name not in names or state_name not in names:
                msg = f"Passed Dataset or DataArray must contain {window_name} and {state_name} in tables.indexes['{index_name}'].names"
                raise ValueError(msg)
        else:
            if window_name not in data.dims or state_name not in data.dims:
                msg = f"Passed Dataset or DataArray must contain {window_name} and {state_name} in dimensions"
                raise ValueError(msg)
            # set index
            data = data.stack({index_name: [window_name, state_name]})  # noqa: PD013

        # indexing dataframe
        frame = (
            data[index_name]  # pyright: ignore[reportIndexIssue]  # py38 only
            .pipe(lambda x: x.copy(data=range(len(x))))
            .to_dataframe()[index_name]
            .reset_index()
            .pipe(
                _filter_min_max_keep_first,
                window_name=window_name,
                state_name=state_name,
                window_index_name=window_index_name,
                check_connected=check_connected,
            )
        )
        # select relevant indices
        data = data.isel({index_name: frame[index_name].to_numpy()})
        # optionally reset window_name
        if reset_window:
            data = data.reset_index(window_name)
            if len(data.indexes[index_name].names) == 1:
                # Only have "state_name"
                data = data.rename({index_name: state_name})
        return data

    # Do calculation
    if isinstance(table, pd.DataFrame):
        return _process_dataframe(table)
    return _process_xarray(table)


# combine on mean
@lru_cache
def _factory_average_updown(
    weight_name: str = "n_trials", down_name: str = "P_down", up_name: str = "P_up"
) -> Callable[[pd.DataFrame], pd.Series[Any]]:
    """Return callable to be used with groupby.apply"""

    columns = [weight_name, down_name, up_name]

    def running_average(g: pd.DataFrame) -> pd.Series[Any]:
        if len(g) == 1:
            return g.iloc[0]

        datas = g[columns].to_numpy()
        out = np.zeros(3, dtype=float)

        for data in datas:
            out[0] += data[0]
            out[1:] += (data[1:] - out[1:]) * data[0] / out[0]

        return pd.Series(out, index=columns)

    return running_average


@docfiller_local
def updown_mean(
    table: pd.DataFrame,
    by: str | Sequence[str] = "state",
    as_index: bool = False,
    weight_name: str = "n_trials",
    down_name: str = "P_down",
    up_name: str = "P_up",
    use_running: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Combine up/down probabilities using weighted average.

    This can be used to splice overlapping windows into a single window, combine across replicate
    simulations, or both.

    Notes
    -----
    If any windows use expanded ensemble (i.e., non-integer ``state`` values),
    then this method should not be used to splice across windows.  Instead, use :func:`keep_first`

    Parameters
    ----------
    table :
        Table containing ``by``, ``up_name``, ``down_name``, and ``weight_name`` columns.
    by :
        Groupby column(s).
    as_index :
        ``as_index`` keyword argument (see :meth:`pandas.DataFrame.groupby`).
    {weight_name}
    {down_name}
    {up_name}
    use_running :
        If False (default), use straight weighted average. If True, use running
        weighted average. The latter can be slower, but numerically stable.
    **kwargs :
        Extra arguments to :meth:`pandas.DataFrame.groupby`

    Returns
    -------
    DataFrame
        Combined transition matrix.
    """

    if use_running:
        columns = [weight_name, down_name, up_name]
        return table.groupby(by, as_index=as_index, **kwargs)[columns].apply(  # type: ignore[arg-type]  # no clue why this is throwing an error
            _factory_average_updown(*columns)
        )

    return (
        table.assign(
            **{
                down_name: lambda x: x[down_name] * x[weight_name],
                up_name: lambda x: x[up_name] * x[weight_name],
            }
        )
        .groupby(by, as_index=as_index, **kwargs)[[weight_name, down_name, up_name]]  # type: ignore[arg-type]
        .sum()
        .assign(
            **{
                down_name: lambda x: x[down_name] / x[weight_name],
                up_name: lambda x: x[up_name] / x[weight_name],
            }
        )
    )


@docfiller_local
def updown_from_collectionmatrix(
    table: pd.DataFrame,
    matrix_names: Iterable[str] = ["c0", "c1", "c2"],
    weight_name: str = "n_trials",
    down_name: str = "P_down",
    up_name: str = "P_up",
) -> pd.DataFrame:
    """
    Add up/down probabilities from collection matrix.

    Parameters
    ----------
    {table_assign}
    matrix_names :
        Column names for collection matrix.
    {weight_name}
    {down_name}
    {up_name}

    Returns
    -------
    DataFrame
        New dataframe with assigned columns.
    """
    matrix_names = list(matrix_names)
    count = table[matrix_names].sum(axis=1)
    return table.assign(
        **{
            weight_name: count,
            down_name: lambda x: x[matrix_names[0]] / count,
            up_name: lambda x: x[matrix_names[-1]] / count,
        }
    )


def _get_delta_lnpi(*, down: NDArray[Any], up: NDArray[Any]) -> NDArray[Any]:
    delta = np.empty_like(down)
    delta[0] = 0.0
    delta[1:] = np.log(up[:-1] / down[1:])
    return delta


@docfiller_local
def delta_lnpi_from_updown(
    down: T_Array,
    up: NDArray[Any] | pd.Series[Any] | xr.DataArray,
    name: str | None = None,
) -> T_Array:
    r"""
    Calculate :math:`\Delta \ln \Pi(N)` from up/down probabilities.

    This assumes ``table`` is sorted by ``state`` value. This function is
    useful if the simulation windows use extended ensemble sampling and have
    non-integer steps in the ``state`` variable. The deltas can be combined
    with :func:`keep_first`, cumalitively summed, then non integer
    states dropped.

    Parameters
    ----------
    {down}
    {up}
    {array_name}

    Returns
    -------
    Series or DataArray or ndarray
        Calculated value of same type as ``down``.
    """

    up_ = (
        up.to_numpy()
        if isinstance(up, (pd.Series, xr.DataArray))
        else cast("NDArray[Any]", up)
    )

    if isinstance(down, pd.Series):
        return pd.Series(
            _get_delta_lnpi(down=down.to_numpy(), up=up_), name=name, index=down.index
        )

    if isinstance(down, xr.DataArray):
        out = down.copy(data=_get_delta_lnpi(down=down.to_numpy(), up=up_))
        if name:
            out = out.rename(name)
        return out

    if isinstance(down, np.ndarray):  # pragma: no branch
        return _get_delta_lnpi(down=down, up=up_)

    msg = f"Unknown type {type(down)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


@docfiller_local
def lnpi_from_updown(
    down: T_Array,
    up: NDArray[Any] | pd.Series[Any] | xr.DataArray,
    # up: T_Array,
    name: str | None = None,
    norm: bool = False,
) -> T_Array:
    r"""
    Calculate :math:`\ln \Pi(N)` from up/down sorted probabilities.

    This assumes ``table`` is sorted by ``state`` value.

    Parameters
    ----------
    {down}
    {up}
    {array_name}
    norm :
        If ``True``, normalize distribution.

    Returns
    -------
    Series or DataArray or ndarray
        Calculated value of same type as ``down``.
    """

    ln_prob: T_Array = delta_lnpi_from_updown(down=down, up=up, name=name).cumsum()
    ln_prob -= ln_prob.max()  # pyright: ignore[reportAssignmentType]
    return normalize_lnpi(ln_prob) if norm else ln_prob


def normalize_lnpi(lnpi: T_Array) -> T_Array:
    r"""Normalize :math:`\ln\Pi` series or array."""
    offset: float = np.log(np.exp(lnpi).sum())
    return lnpi - offset


@docfiller_local
def assign_delta_lnpi_from_updown(
    table: T_Frame,
    up_name: str = "P_up",
    down_name: str = "P_down",
    delta_lnpi_name: str = "delta_lnpi",
) -> T_Frame:
    r"""
    Add :math:`\Delta \ln \Pi(N) = \ln \Pi(N) - \ln \Pi(N-1)` from up/down probabilities.

    This assumes ``table`` is sorted by ``state`` value. This function is
    useful if the simulation windows use extended ensemble sampling and have
    non-integer steps in the ``state`` variable. The deltas can be combined
    with :func:`keep_first`, cumalitively summed, then non integer
    states dropped.

    Parameters
    ----------
    {up_name}
    {down_name}
    delta_lnpi_name :
        Name of output column.

    Returns
    -------
    DataFrame
        Table with ``delta_lnpi_name`` column.
    """

    delta = delta_lnpi_from_updown(up=table[up_name], down=table[down_name])  # type: ignore[arg-type]

    if isinstance(table, pd.DataFrame):
        return table.assign(**{delta_lnpi_name: delta})
    return table.assign({delta_lnpi_name: table[down_name].copy(data=delta)})


@docfiller_local
def assign_lnpi_from_updown(
    table: T_Frame,
    lnpi_name: str = "ln_prob",
    down_name: str = "P_down",
    up_name: str = "P_up",
    norm: bool = True,
) -> T_Frame:
    r"""
    Assign :math:`\ln \Pi(N)` from up/down sorted probabilities.

    This assumes ``table`` is sorted by ``state`` value.

    Parameters
    ----------
    {table_assign}
    {lnpi_name}
    {down_name}
    {up_name}
    use_prod :
        If true (default), calculate from cumulative product (on probability).
        Otherwise, calculate from cumulative sum (on log of probability).
    norm :
        If true (default), normalize distribution.

    Returns
    -------
    DataFrame
        New dataframe with assigned :math:`\ln \Pi`.
    """

    ln_prob = lnpi_from_updown(
        down=table[down_name],  # type: ignore[arg-type]
        up=table[up_name],
        norm=norm,
        name=lnpi_name,
    )

    if isinstance(table, pd.DataFrame):
        return table.assign(**{lnpi_name: ln_prob})
    return table.assign({lnpi_name: table[down_name].copy(data=ln_prob)})
