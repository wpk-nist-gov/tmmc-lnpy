# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
r"""
Routines to combine :math:`\ln \Pi` data (:mod:`~lnpy.combine`)
===============================================================

"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator, TypeVar, cast

import numpy as np
import pandas as pd
from module_utilities.docfiller import DocFiller
from scipy.sparse import coo_array

from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Sequence

    from numpy.typing import NDArray

    T = TypeVar("T", pd.Series[Any], NDArray[Any])


_docstrings_local = r"""
Parameters
----------
lnpi_name :
    Column name corresponding to :math:`\ln \Pi`.
window_name :
    Column name corresponding to "window", i.e., an individual simulation.
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
    :class:`pandas.DataFrame`
check_connected :
    If ``True``, check that all windows form a connected graph.
tables :
    Individual sample windows. If pass in a single
    :class:`~pandas.DataFrame`, it must contain the column ``window_name``.
    Otherwise, the individual frames will be concatenated and the
    ``window_name`` column will be added (or replaced if already present).



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


# * Matrix Equation construction
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

    # .. math::

    #     \min_{C_0, C_1, ..., C_{W-1}} \sum_{\rm{overlap}_m} \in \rm{overlaps}}\, \sum_{N_m, k \in \rm{overlap}_m} [\ln \bar{\Pi}(N_m) - (\ln \Pi_k (N_m) + C_k)]^2


def _create_initial_table(
    tables: pd.DataFrame | Iterable[pd.DataFrame],
    window_name: str,
    window_index_name: str,
) -> pd.DataFrame:
    if isinstance(tables, pd.DataFrame):
        table = tables
        if window_name not in table.columns:
            msg = f"Passed single table must contain {window_name=} column."
            raise ValueError(msg)
        # add in window_index_name
        return _add_window_index(
            table, window_name=window_name, window_index_name=window_index_name
        )

    tables = iter(tables)
    first = next(tables)

    table = pd.concat(
        dict(enumerate(itertools.chain([first], tables))),
        names=[window_index_name, *first.index.names],
    )
    if window_index_name in table.columns:
        table = table.drop(window_index_name, axis=1)
    return table.reset_index(window_index_name)


@docfiller_local
def combine_scaled_lnpi(
    tables: pd.DataFrame | Iterable[pd.DataFrame],
    macrostate_names: str | Iterable[str] = "state",
    lnpi_name: str = "ln_prob",
    window_name: str = "window",
    use_sparse: bool = True,
    check_connected: bool = False,
) -> pd.DataFrame:
    r"""
    Combine multiple windows by scaling each :math:`\ln \Pi`.

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
    """
    window_index_name = "_window_index"
    table = _create_initial_table(
        tables, window_name=window_name, window_index_name=window_index_name
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

    table[lnpi_name] += shift[table[window_index_name].to_numpy()]
    return table.drop(window_index_name, axis=1)


# * Collection matrix
@docfiller_local
def combine_dropfirst(
    tables: pd.DataFrame | Iterable[pd.DataFrame],
    window_name: str = "window",
    state_name: str = "state",
    check_connected: bool = False,
) -> pd.DataFrame:
    """
    Combine windows by dropping first elements that overlap previous window.

    For example, if have two windows `A` and `B` with states `state_A=[0,1,2]`
    and `state_B=[1,2,3]` and observable `x_A(state_A)`, `x_B(state_B)`, then
    the combined result will be `state=[0,1,2,3]`, `x = [x_A(0), x_A(1),
    x_A(2), x_B(3)]`.

    Parameters
    ----------
    {tables}
    {window_name}
    {state_name}
    {check_connected}

    Returns
    -------
    DataFrame
        Combined table.

    Note
    ----
    If there is not expanded ensemble sampling (i.e., non-integer ``state``
    values) in windows, you should prefer using :func:`combine_updown_mean`.
    """

    if isinstance(tables, pd.DataFrame):
        table = tables
        if window_name not in tables.columns:
            msg = f"Passed single table must contain {window_name=} column."
            raise ValueError(msg)
    else:
        import itertools

        tables = iter(tables)
        first = next(tables)

        table = pd.concat(
            dict(enumerate(itertools.chain([first], tables))),
            names=[window_name, *first.index.names],
        )
        if window_name in table.columns:
            table = table.drop(window_name, axis=1)
        table = table.reset_index(window_name)

    window_index_name = "_window_index"
    window_map = (
        table[[window_name, state_name]]
        .groupby(window_name, as_index=False)
        .min()
        .sort_values(state_name)[window_name]
        .pipe(lambda x: pd.Series(range(len(x)), index=x))
    )

    table = table.assign(
        **{window_index_name: window_map[table[window_name]].to_numpy()}
    )
    # return table
    window_max = cast(int, table[window_index_name].iloc[-1])
    if window_max == 0:
        return table

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
def combine_updown_mean(
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
    then this method should not be used to splice across windows.  Instead, use :func:`combine_dropfirst`

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


@docfiller_local
def delta_lnpi_from_updown(
    table: pd.DataFrame,
    up_name: str = "P_up",
    down_name: str = "P_down",
    delta_lnpi_name: str = "delta_lnpi",
) -> pd.DataFrame:
    r"""
    Add :math:`\Delta \ln Pi(N) = \ln Pi(N) - \ln Pi(N-1)` from up/down probabilities.

    This assumes ``table`` is sorted by ``state`` value.

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
    down = table[down_name].to_numpy()
    up = table[up_name].to_numpy()
    delta = np.empty_like(down)
    delta[0] = 0.0
    delta[1:] = np.log(up[:-1] / down[1:])
    return table.assign(**{delta_lnpi_name: delta})


@docfiller_local
def lnpi_from_updown(
    table: pd.DataFrame,
    lnpi_name: str = "ln_prob",
    down_name: str = "P_down",
    up_name: str = "P_up",
    norm: bool = True,
) -> pd.DataFrame:
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
    down = table[down_name].to_numpy()
    up = table[up_name].to_numpy()

    ln_prob = np.empty_like(down)

    ln_prob[0] = 0.0
    ln_prob[1:] = np.log(up[:-1] / down[1:]).cumsum()

    ln_prob -= ln_prob.max()

    return table.assign(**{lnpi_name: normalize_lnpi(ln_prob) if norm else ln_prob})


def normalize_lnpi(lnpi: T) -> T:
    r"""Normalize :math:`\ln\Pi` series or array."""
    offset: float = np.log(np.exp(lnpi).sum())
    return lnpi - offset
