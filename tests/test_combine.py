from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lnpy import combine


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


def test_check_windows_overlap() -> None:
    # no overlaps:
    overlap = pd.DataFrame(
        {
            "n": [],
            "window": [],
        }
    )

    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(3),
            macrostate_names="n",
            window_index_name="window",
            verbose=False,
        )

    overlap = pd.DataFrame({"n": [0, 0, 1, 1, 2, 2], "window": [0, 1, 1, 2, 2, 3]})

    combine.check_windows_overlap(
        overlap,
        windows=range(4),
        macrostate_names="n",
        window_index_name="window",
        verbose=True,
    )

    # missing node
    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(5),
            macrostate_names="n",
            window_index_name="window",
            verbose=True,
        )
    # two graphs
    overlap = pd.DataFrame(
        {
            "n": [0, 0, 1, 1],
            "window": [0, 1, 2, 3],
        }
    )

    with pytest.raises(combine.OverlapError):
        combine.check_windows_overlap(
            overlap,
            windows=range(4),
            macrostate_names="n",
            window_index_name="window",
            verbose=True,
        )


@pytest.fixture()
def table() -> pd.DataFrame:
    x = np.linspace(0, 10)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x),
            "z": np.cos(x),
        }
    )


@pytest.fixture(params=[1, 2, 4, 5])
def dfs(table: pd.DataFrame, request: pytest.FixtureRequest) -> list[pd.DataFrame]:
    n = request.param

    if n == 1:
        return [table]

    split = len(table) // n
    lb = 0
    dfs = []
    for i in range(n + 1):
        ub = lb + split + 1
        dfs.append(table.iloc[lb:ub].assign(y=lambda x: x["y"] + 10 * i))  # noqa: B023

        lb = lb + split
        if lb >= len(table):
            break

    return dfs


def test_dfs(table: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    if len(dfs) == 1:
        pd.testing.assert_frame_equal(table, dfs[0])

    else:
        other = pd.concat(dfs).groupby("x", as_index=False).mean()

        np.testing.assert_allclose(table["x"], other["x"])
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(table["y"], other["y"])


# * combine_scaled
def test_combine_scaled_lnpi_no_overlap(table: pd.DataFrame) -> None:
    mid = len(table) // 2
    # no overlap:
    dfs = [table.iloc[:mid], table.iloc[mid:]]
    with pytest.raises(combine.OverlapError):
        combine.combine_scaled_lnpi(
            dfs,
            macrostate_names="x",
            lnpi_name="y",
        ).groupby("x", as_index=False).mean()


def test_combine_scaled_lnpi_single_table(table: pd.DataFrame) -> None:
    # single table
    dfs = [table]
    new = (
        combine.combine_scaled_lnpi(
            dfs,
            macrostate_names="x",
            lnpi_name="y",
        )
        .groupby("x", as_index=False)
        .mean()
    )
    pd.testing.assert_frame_equal(table, new[table.columns])


mark_connected = pytest.mark.parametrize("check_connected", [False, True])
mark_sparse = pytest.mark.parametrize("use_sparse", [False, True])


@mark_connected
@mark_sparse
def test_combine_scaled_lnpi_split(
    table: pd.DataFrame,
    dfs: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    new = (
        combine.combine_scaled_lnpi(
            dfs,
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])


@mark_connected
@mark_sparse
def test_combine_scaled_lnpi_split_other(
    table: pd.DataFrame,
    dfs: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    # with window name already there
    new = (
        combine.combine_scaled_lnpi(
            [x.assign(window="hello", other=1, _window_index="there") for x in dfs],
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .drop("window", axis=1)
        .groupby("x", as_index=False)
        .mean()
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])


@mark_connected
@mark_sparse
def test_combine_scaled_lnpi_split_single_table(
    table: pd.DataFrame,
    dfs: list[pd.DataFrame],
    check_connected: bool,
    use_sparse: bool,
) -> None:
    # already formed single table:
    new = (
        combine.combine_scaled_lnpi(
            pd.concat((x.assign(window=i) for i, x in enumerate(dfs))),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])

    # wrong name:
    with pytest.raises(ValueError, match=r".* single table must contain .*"):
        combine.combine_scaled_lnpi(
            pd.concat((x.assign(window_wrong=i) for i, x in enumerate(dfs))),
            macrostate_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )


# * combine.combine_dropfirst
def test_combine_dropfirst_no_overlap(table: pd.DataFrame) -> None:
    mid = len(table) // 2
    dfs = [table.iloc[:mid], table.iloc[mid:]]
    with pytest.raises(combine.OverlapError):
        combine.combine_dropfirst(dfs, state_name="x", check_connected=True)


def test_combine_dropfirst_single_table(table: pd.DataFrame) -> None:
    new = combine.combine_dropfirst([table], state_name="x")
    pd.testing.assert_frame_equal(table, new[table.columns])


def test_combine_dropfirst_split(table: pd.DataFrame, dfs: list[pd.DataFrame]) -> None:
    new = combine.combine_dropfirst(dfs, state_name="x")
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # test odd window names:
    new = combine.combine_dropfirst(
        [x.assign(window="hello", other=1, _window_index="there") for x in dfs],
        state_name="x",
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # using single table:
    new = combine.combine_dropfirst(
        pd.concat((x.assign(window=i) for i, x in enumerate(dfs))), state_name="x"
    )
    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["z"], new["z"])

    # wrong name
    with pytest.raises(ValueError, match=r".* single table must contain .*"):
        combine.combine_dropfirst(
            pd.concat((x.assign(window_wrong=i) for i, x in enumerate(dfs))),
            state_name="x",
        )


# * Utilities
@pytest.mark.parametrize("use_running", [True, False])
@pytest.mark.parametrize("as_index", [True, False])
def test_combine_updown_mean(
    as_index: bool, use_running: bool, rng: np.random.Generator
) -> None:
    nstate = 10
    nwindow = 2
    weight, down, up = rng.random((3, nwindow, nstate))

    df_updown = pd.DataFrame(
        {
            "window": np.repeat(range(nwindow), nstate),
            "state": np.tile(range(nstate), nwindow),
            "n_trials": weight.reshape(-1),
            "P_down": down.reshape(-1),
            "P_up": up.reshape(-1),
        }
    )

    # add a single value
    df_add = pd.DataFrame(
        [
            {
                "window": nwindow,
                "state": nstate,
                "n_trials": rng.random(),
                "P_down": rng.random(),
                "P_up": rng.random(),
            }
        ]
    )
    df_total = pd.concat((df_updown, df_add))

    out = combine.combine_updown_mean(
        df_total, by="state", as_index=as_index, use_running=use_running
    )

    out_head = out.iloc[:-1]
    np.testing.assert_allclose(out_head["n_trials"], weight.sum(0))
    np.testing.assert_allclose(
        out_head["P_down"], np.average(down, axis=0, weights=weight)
    )
    np.testing.assert_allclose(out_head["P_up"], np.average(up, axis=0, weights=weight))

    out_tail = out.iloc[-1]
    for k in ["n_trials", "P_down", "P_up"]:
        np.testing.assert_allclose(out_tail[k], df_add.iloc[-1][k])


def test_updown_from_collectionmatrix(rng: np.random.Generator) -> None:
    c = rng.random((10, 3))

    table = pd.DataFrame(
        {
            "n_trials": c.sum(-1),
            "P_down": c[:, 0] / c.sum(-1),
            "P_up": c[:, -1] / c.sum(-1),
        }
    )

    out = combine.updown_from_collectionmatrix(
        pd.DataFrame(c, columns=["c0", "c1", "c2"])
    )

    pd.testing.assert_frame_equal(table, out[table.columns])


@pytest.fixture()
def table_updown(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(rng.random((10, 3)), columns=["n_trials", "P_down", "P_up"])


def test_delta_lnpi_from_updown(table_updown: pd.DataFrame) -> None:
    out = combine.delta_lnpi_from_updown(table_updown)

    delta_lnpi = (
        np.log(table_updown["P_up"].shift(1) / table_updown["P_down"]).fillna(0.0)  # pyright: ignore[reportAttributeAccessIssue]
    )
    np.testing.assert_allclose(delta_lnpi, out["delta_lnpi"])


def test_lnpi_from_updown(table_updown: pd.DataFrame) -> None:
    table = (
        combine.delta_lnpi_from_updown(table_updown)
        .assign(ln_prob=lambda x: x["delta_lnpi"].cumsum())
        .assign(ln_prob=lambda x: x["ln_prob"] - x["ln_prob"].max())
    )
    out = combine.lnpi_from_updown(table_updown, norm=False)
    np.testing.assert_allclose(out["ln_prob"], table["ln_prob"])

    out = combine.lnpi_from_updown(table_updown, norm=True)
    table = table.assign(
        ln_prob=lambda x: x["ln_prob"] - np.log(np.exp(x["ln_prob"]).sum())
    )
    np.testing.assert_allclose(out["ln_prob"], table["ln_prob"])


# @mark_connected
# @mark_sparse
# def test_split_mult(table: pd.DataFrame, check_connected: bool, use_sparse: bool) -> None:

#     # multiple splits:
#     n = 5
#     split = len(table) // n
#     lb = 0

#     dfs = []
#     for i in range(n + 1):
#         ub = lb + split + 1
#         dfs.append(table.iloc[lb:ub].assign(y=lambda x: x["y"] + 10 * i))

#         lb = lb + split
#         if lb >= len(table):
#             break

#     new = (
#         combine.combine_scaled_lnpi(
#             dfs,
#             macrostate_names="x",
#             lnpi_name="y",
#             use_sparse=use_sparse,
#             check_connected=check_connected,
#         )
#         .groupby("x", as_index=False)
#         .mean()
#     )

#     np.testing.assert_allclose(table["x"], new["x"])
#     np.testing.assert_allclose(table["y"], new["y"])
