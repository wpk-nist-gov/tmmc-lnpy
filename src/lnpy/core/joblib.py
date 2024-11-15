"""Routines interacting with joblib"""

from __future__ import annotations

from functools import lru_cache
from itertools import starmap
from typing import TYPE_CHECKING

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Any

    from .typing import R


@lru_cache
def _get_joblib() -> Any:
    try:
        import joblib
    except ImportError:
        joblib = None
    return joblib


def _use_joblib(
    items: Sequence[Any],
    len_key: str,
    use_joblib: bool = True,
    total: int | None = None,
) -> bool:
    joblib = _get_joblib()

    if use_joblib and joblib and OPTIONS["joblib_use"]:
        if total is None:
            total = len(items)

        return total >= OPTIONS[len_key]  # type: ignore[no-any-return,literal-required]
    return False


def _parallel(seq: Iterable[Any]) -> list[Any]:
    joblib = _get_joblib()

    return joblib.Parallel(  # type: ignore[no-any-return]
        n_jobs=OPTIONS["joblib_n_jobs"],
        backend=OPTIONS["joblib_backend"],
        **OPTIONS["joblib_kws"],
    )(seq)


def parallel_map_build(
    func: Callable[..., R], items: Iterable[Any], *args: Any, **kwargs: Any
) -> list[R]:
    joblib = _get_joblib()

    items = tuple(items)

    if _use_joblib(items, "joblib_len_build"):
        return _parallel(joblib.delayed(func)(x, *args, **kwargs) for x in items)
    return [func(x, *args, **kwargs) for x in items]


def _func_call(x: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    return x(*args, **kwargs)


def parallel_map_call(
    items: Sequence[Callable[..., Any]],
    use_joblib: bool,  # noqa: ARG001
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    joblib = _get_joblib()
    if _use_joblib(items, "joblib_len_calc"):
        return _parallel(joblib.delayed(_func_call)(x, *args, **kwargs) for x in items)
    return [x(*args, **kwargs) for x in items]


def parallel_map_attr(attr: str, use_joblib: bool, items: Sequence[Any]) -> list[Any]:  # noqa: ARG001
    from operator import attrgetter

    joblib = _get_joblib()
    func = attrgetter(attr)

    if _use_joblib(items, "joblib_len_calc"):
        return _parallel(joblib.delayed(func)(x) for x in items)
    return [func(x) for x in items]


def parallel_map_func_starargs(
    func: Callable[..., R],
    use_joblib: bool,  # noqa: ARG001
    items: Iterable[Any],
    total: int | None = None,
) -> list[R]:
    joblib = _get_joblib()

    items = tuple(items)

    if _use_joblib(items, "joblib_len_calc", total=total):
        return _parallel(starmap(joblib.delayed(func), items))
    return list(starmap(func, items))
