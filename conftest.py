"""Config for doctests and test collection"""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace) -> None:  # noqa: ARG001, D103
    import numpy as np

    np.set_printoptions(precision=4)


def pytest_ignore_collect(collection_path) -> None:  # noqa: ARG001, D103
    return False
