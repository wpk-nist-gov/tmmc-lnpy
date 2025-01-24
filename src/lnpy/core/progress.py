"""Routines to work with progress bar."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType
    from typing import Any

    from .typing_compat import TypeVar

    T = TypeVar("T")


# * TQDM setup ----------------------------------------------------------------
@lru_cache
def _get_tqdm() -> ModuleType | None:
    try:
        import tqdm
    except ImportError:
        return None
    else:
        return tqdm


@lru_cache
def _get_tqdm_default() -> Callable[..., Any]:
    tqdm_ = _get_tqdm()
    if tqdm_:
        try:
            from IPython.core.getipython import (  # pyright: ignore[reportMissingImports]
                get_ipython,
            )

            p = get_ipython()  # type: ignore[no-untyped-call, unused-ignore]
            if p is not None and p.has_trait("kernel"):
                from tqdm.notebook import tqdm as tqdm_default

                return tqdm_default
            return cast("Callable[..., Any]", tqdm_.tqdm)
        except ImportError:
            return cast("Callable[..., Any]", tqdm_.tqdm)
    else:

        def wrapper(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:  # noqa: ARG001
            return seq

        return wrapper


def tqdm(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:
    opt = OPTIONS["tqdm_bar"]
    tqdm_ = _get_tqdm()
    if tqdm_:
        if opt == "text":
            func = tqdm_.tqdm
        elif opt == "notebook":
            func = tqdm_.tqdm_notebook
        else:
            func = _get_tqdm_default()
    else:
        func = _get_tqdm_default()
    return cast("Iterable[T]", func(seq, *args, **kwargs))


def get_tqdm(
    seq: Iterable[T], len_min: str | int, leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    n = kwargs.get("total")
    tqdm_ = _get_tqdm()

    if isinstance(len_min, str):
        len_min = OPTIONS[len_min]  # type: ignore[literal-required]

        if not isinstance(len_min, int):
            msg = f"{type(len_min)=} must be an int or a string in options."
            raise TypeError(msg)

    if n is None:
        seq = tuple(seq)
        n = len(seq)

    if tqdm_ and OPTIONS["tqdm_use"] and n >= len_min:
        if leave is None:
            leave = OPTIONS["tqdm_leave"]
        seq = tqdm(seq, leave=leave, **kwargs)
    return seq


def get_tqdm_calc(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_calc", leave=leave, **kwargs)


def get_tqdm_build(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_build", leave=leave, **kwargs)
