"""Overlap routines"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_array

from lnpy.core.validate import is_dataarray, is_series, validate_str_or_iterable
from lnpy.core.xr_utils import factory_apply_ufunc_kwargs, select_axis_dim

from ._docfiller import docfiller_local
from .grouper import factory_indexed_grouper

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Any

    from numpy.typing import NDArray

    from lnpy.core.typing import (
        ApplyUFuncKwargs,
        DimsReduce,
        FactoryIndexedGrouperTypes,
        FrameOrDataT,
        GenArrayOrSeriesT,
        KeepAttrs,
        NDArrayAny,
    )


class OverlapError(ValueError):
    """Specific error for missing overlaps."""


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
    macrostate_names = validate_str_or_iterable(macrostate_names)
    overlap_table = overlap_table[[window_index_name, *macrostate_names]]

    x: pd.DataFrame = (
        overlap_table.merge(
            overlap_table, on=macrostate_names, how="outer", suffixes=("", "_nebr")
        )
        .drop(macrostate_names, axis=1)
        .query(f"{window_index_name} < {window_index_name}_nebr")
        .drop_duplicates()
    )

    graph = _build_graph(nodes=windows, edges=x.to_numpy())  # to_numpy())

    components = list(_connected_components(graph))

    if len(components) != 1:
        msg = "Disconnected graph."
        if verbose:
            for subgraph in components:
                msg = f"{msg}\ngraph: {set(map(int, subgraph))}"
        raise OverlapError(msg)


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


@np.vectorize(signature="(n), (n), (n, d), (), () -> (n)")  # type: ignore[call-arg]
def _shift_lnpi_windows(
    lnpi: NDArrayAny,
    windows: NDArrayAny,
    macrostate: NDArrayAny,
    use_sparse: bool = True,
    check_connected: bool = True,
) -> NDArrayAny:
    # first get unique windows
    window_codes, _ = pd.factorize(windows, sort=False)
    window_max = window_codes.max()

    if window_max == 0:
        return lnpi

    if macrostate.ndim == 1:
        macrostate = macrostate.reshape(-1, 1)
    macrostate_names = [f"macrostate_{i}" for i in range(macrostate.shape[1])]

    # construct overlap table
    window_index_name = "_window_index"
    lnpi_name = "lnpi"

    table = pd.DataFrame(
        {
            lnpi_name: lnpi,
            window_index_name: window_codes,
        }
    )
    table[macrostate_names] = macrostate

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

    return lnpi + shift[window_codes]  # type: ignore[no-any-return]


def _shift_lnpi_windows_indexed(
    lnpi: NDArrayAny,
    windows: NDArrayAny,
    macrostate: NDArrayAny,
    index: Sequence[int],
    group_start: Sequence[int],
    group_end: Sequence[int],
    use_sparse: bool = True,
    check_connected: bool = True,
) -> NDArrayAny:
    # _shift_lnpi_windows applied over groups

    out = []
    for start, end in zip(group_start, group_end):
        idx = index[start:end]

        out.append(
            _shift_lnpi_windows(
                np.take(lnpi, idx, axis=-1),
                np.take(windows, idx, axis=-1),
                np.take(macrostate, idx, axis=-2),
                use_sparse,
                check_connected,
            )
        )
    return np.concatenate(out, axis=-1)


@docfiller_local
def shift_lnpi_windows(
    lnpi: GenArrayOrSeriesT,
    window: GenArrayOrSeriesT,
    *macrostate: GenArrayOrSeriesT,
    use_sparse: bool = False,
    check_overlap: bool = False,
    dim: DimsReduce | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> GenArrayOrSeriesT:
    if is_series(lnpi):
        return pd.Series(
            shift_lnpi_windows(
                *(a.to_numpy() for a in chain((lnpi, window), macrostate)),  # type: ignore[arg-type]
                use_sparse=use_sparse,
                check_overlap=check_overlap,
            ),
            index=lnpi.index,
        )

    if is_dataarray(lnpi):
        _, dim = select_axis_dim(lnpi, -1, dim)
        return xr.apply_ufunc(  # type: ignore[no-any-return]
            shift_lnpi_windows,
            lnpi,
            window,
            *macrostate,
            kwargs={
                "use_sparse": use_sparse,
                "check_overlap": check_overlap,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
        )

    return cast(  # pyright: ignore[reportReturnType]
        "NDArrayAny",
        _shift_lnpi_windows(
            lnpi,
            window,
            np.stack(macrostate, axis=-1),
            use_sparse,
            check_overlap,
        ),
    )


@docfiller_local
def shift_lnpi_windows_indexed(
    lnpi: GenArrayOrSeriesT,
    window: GenArrayOrSeriesT,
    *macrostate: GenArrayOrSeriesT,
    grouper: FactoryIndexedGrouperTypes | None = None,
    use_sparse: bool = False,
    check_overlap: bool = False,
    dim: DimsReduce | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> GenArrayOrSeriesT:
    grouper = factory_indexed_grouper(grouper, data=lnpi, dim=dim, axis=-1)

    if is_series(lnpi):
        return pd.Series(
            shift_lnpi_windows_indexed(
                *(a.to_numpy() for a in chain((lnpi, window), macrostate)),  # type: ignore[arg-type]
                grouper=grouper,
                use_sparse=use_sparse,
                check_overlap=check_overlap,
            ),
            index=lnpi.index,
        )

    if is_dataarray(lnpi):
        _, dim = select_axis_dim(lnpi, -1, dim)
        return xr.apply_ufunc(  # type: ignore[no-any-return]
            shift_lnpi_windows_indexed,
            lnpi,
            window,
            *macrostate,
            kwargs={
                "use_sparse": use_sparse,
                "check_overlap": check_overlap,
                "grouper": grouper,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
        )

    return cast(  # pyright: ignore[reportReturnType]
        "NDArrayAny",
        _shift_lnpi_windows_indexed(
            lnpi,
            window,  # pyright: ignore[reportArgumentType]
            np.stack(macrostate, axis=-1),
            grouper.index,
            grouper.start,
            grouper.end,
            use_sparse,
            check_overlap,
        ),
    )


def assign_shift_lnpi_windows_indexed(
    table: FrameOrDataT,
    *,
    window_name: str = "window",
    macrostate_names: str | Iterable[str] = "state",
    lnpi_name: str = "ln_prob",
    grouper: FactoryIndexedGrouperTypes | None = None,
    use_sparse: bool = False,
    check_overlap: bool = False,
    dim: DimsReduce | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> FrameOrDataT:
    r"""Create new object with shifted :math:`\ln \Pi(N)`"""

    grouper = factory_indexed_grouper(grouper, data=table, dim=dim, axis=-1)

    out = shift_lnpi_windows_indexed(
        table if is_dataarray(table) else table[lnpi_name],
        table[window_name],
        *(table[k] for k in validate_str_or_iterable(macrostate_names)),
        grouper=grouper,
        use_sparse=use_sparse,
        check_overlap=check_overlap,
        dim=dim,
        keep_attrs=keep_attrs,
        apply_ufunc_kwargs=apply_ufunc_kwargs,
    )

    if is_dataarray(table):
        return out  # pyright: ignore[reportReturnType]
    return table.assign(**{lnpi_name: out})  # type: ignore[arg-type]
