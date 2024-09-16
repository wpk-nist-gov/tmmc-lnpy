"""Handle typing compatibility issues."""

import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec, TypeAlias
else:
    from typing import Concatenate, ParamSpec, TypeAlias

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

IndexAny: TypeAlias = "pd.Index[Any]"  # type: ignore[misc, unused-ignore]  # get pd.Index working for python 3.8


__all__ = [
    "Concatenate",
    "IndexAny",
    "ParamSpec",
    "Self",
    "TypeAlias",
]
