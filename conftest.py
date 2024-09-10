"""Config for doctests and test collection"""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace) -> None:  # noqa: ARG001
    import numpy as np

    np.set_printoptions(precision=4)


def pytest_ignore_collect(collection_path, path, config) -> None:  # noqa: ARG001
    import sys

    if sys.version_info < (3, 9):
        return "tests/test_" not in str(collection_path)

    return False
