import numpy as np
import pandas as pd
import pytest

from lnpy.splice import OverlapError, check_windows_overlap, splice


def test_check_windows_overlap() -> None:
    # no overlaps:
    overlap = pd.DataFrame(
        {
            "n": [],
            "window": [],
        }
    )

    with pytest.raises(OverlapError):
        check_windows_overlap(
            overlap,
            windows=range(3),
            state_names="n",
            window_name="window",
            verbose=False,
        )

    overlap = pd.DataFrame({"n": [0, 0, 1, 1, 2, 2], "window": [0, 1, 1, 2, 2, 3]})

    assert (
        check_windows_overlap(
            overlap,
            windows=range(4),
            state_names="n",
            window_name="window",
            verbose=True,
        )
        is None
    )

    # missing node
    with pytest.raises(OverlapError):
        check_windows_overlap(
            overlap,
            windows=range(5),
            state_names="n",
            window_name="window",
            verbose=True,
        )
    # two graphs
    overlap = pd.DataFrame(
        {
            "n": [0, 0, 1, 1],
            "window": [0, 1, 2, 3],
        }
    )

    with pytest.raises(OverlapError):
        check_windows_overlap(
            overlap,
            windows=range(4),
            state_names="n",
            window_name="window",
            verbose=True,
        )


@pytest.mark.parametrize("check_connected", [False, True])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_simple(check_connected: bool, use_sparse: bool) -> None:
    x = np.linspace(0, 10)

    table = pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x),
        }
    )

    mid = len(table) // 2

    # no overlap:
    dfs = [table.iloc[:mid], table.iloc[mid:]]
    with pytest.raises(OverlapError):
        splice(
            dfs,
            state_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        ).groupby("x", as_index=False).mean()

    # single table
    dfs = [table]
    new = (
        splice(
            dfs,
            state_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    pd.testing.assert_frame_equal(table, new[["x", "y"]])

    dfs = [table.iloc[: mid + 1], table.iloc[mid:].assign(y=lambda x: x["y"] + 10)]
    new = (
        splice(
            dfs,
            state_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])

    # with window name already there
    dfs = [x.assign(window="hello", other=1) for x in dfs]
    new = (
        splice(
            dfs,
            state_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])

    # multiple splits:
    n = 5
    split = len(table) // n
    lb = 0

    dfs = []
    for i in range(n + 1):
        ub = lb + split + 1
        dfs.append(table.iloc[lb:ub].assign(y=lambda x: x["y"] + 10 * i))  # noqa: B023

        lb = lb + split
        if lb >= len(table):
            break

    new = (
        splice(
            dfs,
            state_names="x",
            lnpi_name="y",
            use_sparse=use_sparse,
            check_connected=check_connected,
        )
        .groupby("x", as_index=False)
        .mean()
    )

    np.testing.assert_allclose(table["x"], new["x"])
    np.testing.assert_allclose(table["y"], new["y"])
