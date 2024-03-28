# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
r"""
Routines to splice :math:`\ln \Pi` data (:mod:`~lnpy.splice`)
=============================================================

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

if TYPE_CHECKING:
    from typing import Any, Iterable

    from numpy.typing import NDArray


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
    state_names: str | Iterable[str] = "state",
    window_name: str = "window",
    verbose: bool = True,
) -> None:
    """
    Check that windows overlap_table into a connected graph.

    Parameters
    ----------
    overlap_table: pd.DataFrame
        Frame should contain columns ``state_names`` and ``window_name``
        and rows corresponding to overlaps.

    Raises
    ------
    OverlapError
        If the overlaps do not form a connected graph, then raise a ``OverlapError``.
    """
    state_names = _str_or_iterable(state_names)
    overlap_table = overlap_table[[window_name, *state_names]]

    x: pd.DataFrame = (
        overlap_table.merge(
            overlap_table, on=state_names, how="outer", suffixes=["", "_nebr"]
        )
        .drop(state_names, axis=1)
        .query(f"{window_name} < {window_name}_nebr")
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
def _create_overlap_table(
    table: pd.DataFrame,
    window_name: str,
    window_max: int,
    state_names: list[str],
    lnpi_name: str,
    check_connected: bool = True,
) -> pd.DataFrame:
    overlap_table = table.loc[
        table[state_names].duplicated(keep=False),
        [window_name, *state_names, lnpi_name],
    ]

    if len(overlap_table) == 0:
        msg = "No overlaps with multiple windows"
        raise OverlapError(msg)

    if check_connected:
        check_windows_overlap(
            windows=range(window_max + 1),
            overlap_table=overlap_table,
            window_name=window_name,
            state_names=state_names,
        )
    return overlap_table


def _create_overlap_total_table(
    overlap_table: pd.DataFrame,
    window_name: str,
    window_max: int,
    state_names: list[str],
    lnpi_name: str,
) -> pd.DataFrame:
    group = overlap_table.groupby(state_names, as_index=False)[lnpi_name]
    return (
        overlap_table.assign(
            total=group.transform("sum"),
            count=group.transform("count"),
        )
        # limit to only window < window_max.
        # Equivalent to not shifting last window.
        .query(f"{window_name} < {window_max}")
        # assign eq_idx
        .assign(eq_idx=lambda x: range(len(x)))
    )


def _create_overlap_outer_table(
    overlap_total_table: pd.DataFrame,
    state_names: list[str],
    window_name: str,
) -> pd.DataFrame:
    return overlap_total_table[[*state_names, "eq_idx"]].merge(
        overlap_total_table[[*state_names, window_name]],
        on=state_names,
        how="outer",
    )


def _create_lhs_matrix_sparse(
    overlap_total_table: pd.DataFrame,
    overlap_outer_table: pd.DataFrame,
    window_name: str,
    window_max: int,
) -> coo_array:
    coeff_table = pd.concat(
        (
            overlap_total_table[["eq_idx", window_name, "count"]],
            overlap_outer_table[["eq_idx", window_name]].assign(count=-1),
        )
    )

    return coo_array(
        (
            coeff_table["count"],
            (coeff_table["eq_idx"], coeff_table[window_name]),
        ),
        shape=(len(overlap_total_table), window_max),
    )


def _create_lhs_matrix_numpy(
    overlap_total_table: pd.DataFrame,
    overlap_outer_table: pd.DataFrame,
    window_name: str,
    window_max: int,
) -> NDArray[Any]:
    overlap_outer_table = (
        overlap_outer_table[["eq_idx", window_name]]
        .assign(count=1)
        .groupby(["eq_idx", window_name], as_index=False)
        .sum()
    )
    a = np.zeros((len(overlap_total_table), window_max))
    a[overlap_total_table["eq_idx"], overlap_total_table["window"]] = (
        overlap_total_table["count"]
    )
    a[overlap_outer_table["eq_idx"], overlap_outer_table["window"]] -= (
        overlap_outer_table["count"]
    )
    return a


def splice(
    tables: Iterable[pd.DataFrame],
    state_names: str | Iterable[str] = "state",
    lnpi_name: str = "ln_prob",
    window_name: str = "window",
    use_sparse: bool = True,
    check_connected: bool = False,
) -> pd.DataFrame:
    r"""
    Splice multiple tmmc scans together in one pass using least squares.

    This performs least squares on the problem:

    .. math::

        \sum_{overlaps} \sum_{N_m, k \in \rm{overlap}} [\ln \bar{\Pi}(N_m) - (\ln \Pi_k (N_m) + C_k)]^2

    where :math:`\Pi_k` is the kth sample, :math:`C_k` is the to be determined shift for each sample, and :math:`\ln \bar{Pi}`
    is the to be determined average value.

    This can be reduced to a matrix problem of the form:

    .. math::

        S C_j - \sum_{k \in \rm{overlap}_m} C_k = - (S \ln \Pi_j(N_m)  - \sum_{k \in \rm{overlap}_m} \ln Pi_k(N_m))

    the sum runs over all samples with overlap at state :math:`N_m`, :math:`S` is the number of such overlaps
    (i.e., :math:`S = \sum_{k \in \rm{overlap}_m 1`).  There are such equations for all `j \in \rm{overlap}_m`.



    Parameters
    ----------
    tables : iterable of pd.DataFrame
        Individual sample windows.
    state_names : str | Iterable[str]
        Column name corresponding to a single state.  For example,
        for single component, this would be something like ``'n'``.  For binary system,
        this would be something like ``['n_0', 'n_1']``.
    lnpi_name : str
        Name of column in ``tables`` corresponding to :math:`\ln \Pi.`
    window_name : str, default="window"
        Name of column with will keep track of the window index.
    use_sparse : bool, default=True
        Use sparse matrix in matrix equation.  This is often faster than using a :class:`numpy.ndarray`.
    check_connected : bool, default=False
        If ``True``, check that all windows form a connected graph.

    Returns
    -------
    pd.DataFrame
        DataFrame of all data.  The frame is not yet averaged over ``state_names``.
    """
    table: pd.DataFrame = pd.concat(dict(enumerate(tables)), names=[window_name, None])
    if window_name in table.columns:
        table = table.drop(window_name, axis=1)
    table = table.reset_index(window_name)

    window_max = cast(int, table[window_name].iloc[-1])
    if window_max == 0:
        return table

    state_names = _str_or_iterable(state_names)
    overlap_total_table = _create_overlap_total_table(
        overlap_table=_create_overlap_table(
            table,
            window_name=window_name,
            window_max=window_max,
            state_names=state_names,
            lnpi_name=lnpi_name,
            check_connected=check_connected,
        ),
        window_name=window_name,
        window_max=window_max,
        state_names=state_names,
        lnpi_name=lnpi_name,
    )

    b: NDArray[Any] = overlap_total_table.pipe(
        lambda x: -x["count"] * x[lnpi_name] + x["total"]
    ).to_numpy()

    overlap_outer_table = _create_overlap_outer_table(
        overlap_total_table=overlap_total_table,
        state_names=state_names,
        window_name=window_name,
    )

    if use_sparse:
        a = _create_lhs_matrix_sparse(
            overlap_total_table=overlap_total_table,
            overlap_outer_table=overlap_outer_table,
            window_name=window_name,
            window_max=window_max,
        )

        lhs = (a.T @ a).toarray()
        rhs = a.T @ b

    else:
        a = _create_lhs_matrix_numpy(
            overlap_total_table=overlap_total_table,
            overlap_outer_table=overlap_outer_table,
            window_name=window_name,
            window_max=window_max,
        )

        lhs = a.T @ a
        rhs = a.T @ b

    shift = np.zeros(window_max + 1)
    shift[:-1] = np.linalg.solve(lhs, rhs)
    shift -= shift[0]

    table[lnpi_name] += shift[table[window_name].to_numpy()]
    return table
