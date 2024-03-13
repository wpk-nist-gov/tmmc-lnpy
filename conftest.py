"""Config for doctests and test collection"""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace) -> None:  # noqa: ARG001
    import numpy as np
    # import pandas as pd
    # import xarray as xr

    # doctest_namespace["np"] = np
    # doctest_namespace["pd"] = pd
    # doctest_namespace["xr"] = xr

    np.set_printoptions(precision=4)


def pytest_ignore_collect(collection_path, path, config) -> None:  # noqa: ARG001
    import sys

    if sys.version_info < (3, 9):
        if "tests/test_" in str(collection_path):
            return False
        return True

    return False
