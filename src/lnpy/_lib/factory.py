"""Factory numba functions."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from typing import Any, Protocol

    from numpy.typing import ArrayLike

    from lnpy.core.typing import NDArrayAny, NDArrayInt

    class DeltalnPiFromUpDown(Protocol):
        def __call__(
            self,
            down: ArrayLike,
            up: ArrayLike,
            index: ArrayLike,
            group_start: ArrayLike,
            group_end: ArrayLike,
            /,
            **kwargs: Any,
        ) -> NDArrayAny: ...

    class IndexedFunc(Protocol):
        def __call__(
            self,
            delta_lnpi: ArrayLike,
            index: ArrayLike,
            group_start: ArrayLike,
            group_end: ArrayLike,
            /,
            **kwargs: Any,
        ) -> NDArrayAny: ...

    class StateMaxWindow(Protocol):
        def __call__(
            self,
            state: ArrayLike,
            window_index: ArrayLike,
            window_start: ArrayLike,
            window_end: ArrayLike,
            /,
            **kwargs: Any,
        ) -> NDArrayInt: ...

    class KeepFirstIndexer(Protocol):
        def __call__(
            self,
            state: ArrayLike,
            state_min: ArrayLike,
            window_index: ArrayLike,
            window_start: ArrayLike,
            window_end: ArrayLike,
            rec_index: ArrayLike,
            rec_start: ArrayLike,
            rec_end: ArrayLike,
            /,
            **kwargs: Any,
        ) -> tuple[NDArrayInt, int]: ...


@lru_cache
def _safe_threadpool() -> bool:
    from .decorators import is_in_unsafe_thread_pool

    return not is_in_unsafe_thread_pool()


def supports_parallel() -> bool:
    """
    Checks if system supports parallel numba functions.

    If an unsafe thread pool is detected, return ``False``.

    Returns
    -------
    bool :
        ``True`` if supports parallel.  ``False`` otherwise.
    """
    return OPTIONS["joblib_use"] and _safe_threadpool()


def parallel_heuristic(
    parallel: bool | None,
    size: int | None = None,
    cutoff: int = 1_000,
) -> bool:
    """Default parallel."""
    if parallel is not None:
        return parallel and supports_parallel()
    if size is None or not supports_parallel():
        return False
    return size > cutoff


def factory_delta_lnpi_from_updown(parallel: bool) -> DeltalnPiFromUpDown:
    if parallel:
        from .combine_parallel import delta_lnpi_from_updown
    else:
        from .combine import delta_lnpi_from_updown

    return cast("DeltalnPiFromUpDown", delta_lnpi_from_updown)


def factory_lnpi_from_delta_lnpi(parallel: bool) -> IndexedFunc:
    if parallel:
        from .combine_parallel import lnpi_from_delta_lnpi
    else:
        from .combine import lnpi_from_delta_lnpi

    return cast("IndexedFunc", lnpi_from_delta_lnpi)


def factory_normalize_lnpi(parallel: bool) -> IndexedFunc:
    if parallel:
        from .combine_parallel import normalize_lnpi
    else:
        from .combine import normalize_lnpi

    return cast("IndexedFunc", normalize_lnpi)


def factory_state_max(parallel: bool) -> StateMaxWindow:
    if parallel:
        from .combine_parallel import state_max_window
    else:
        from .combine import state_max_window

    return cast("StateMaxWindow", state_max_window)


def factory_keep_first_indexer(parallel: bool) -> KeepFirstIndexer:
    if parallel:
        from .combine_parallel import keep_first_indexer
    else:
        from .combine import keep_first_indexer

    return cast("KeepFirstIndexer", keep_first_indexer)
