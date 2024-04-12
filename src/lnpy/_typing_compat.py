"""Handle typing compatibility issues."""

import sys

import pandas as pd

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec, TypeAlias
else:
    from typing import Concatenate, ParamSpec, TypeAlias

if sys.version_info < (3, 9):
    IndexAny = pd.Index
else:
    from typing import Any

    IndexAny: TypeAlias = "pd.Index[Any]"  # type: ignore[misc, unused-ignore]  # get pd.Index working for python 3.8


__all__ = [
    "Concatenate",
    "IndexAny",
    "ParamSpec",
    "Self",
    "TypeAlias",
]
