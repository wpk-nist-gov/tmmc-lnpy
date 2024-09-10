"""
Define accessor routines.

This is inspired by xarray accessors.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Generic, overload

from module_utilities import cached
from module_utilities.typing import C_prop, R, S

if TYPE_CHECKING:
    from typing import Literal

    from module_utilities.cached import CachedProperty


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _CachedAccessorSingle(Generic[S, R]):
    """
    Custom property-like object (descriptor).
    this gets created once on call
    """

    def __init__(self, name: str, accessor: C_prop[S, R]) -> None:
        self.__doc__ = accessor.__doc__

        self._name = name
        self._accessor = accessor

    @overload
    def __get__(self, obj: None, cls: type[S] | None = None) -> C_prop[S, R]: ...

    @overload
    def __get__(self, obj: S, cls: type[S] | None = None) -> R: ...

    def __get__(self, obj: S | None, cls: type[S] | None = None) -> C_prop[S, R] | R:
        if obj is None:
            return self._accessor
        try:
            accessor_obj = self._accessor(obj)
        except AttributeError as err:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            msg = f"error initializing {self._name!r} accessor."
            raise RuntimeError(msg) from err
        # Replace the property with the accessor object. Inspired by:
        # http://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # AttrAccessMixin.
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _cachedaccessorcleared(name: str, accessor: C_prop[S, R]) -> CachedProperty[S, R]:
    """
    Wrap accessor in a cached property

    This creates an property in the parent class with name `name`
    from `accessor(self)`.
    This only gets initialized when called

    Notes
    -----
    This gets deleted if self._cache is cleared.
    If you only want to create it once, then use the Single access wrapper below
    """

    return cached.prop(key=name)(accessor)


@overload
def _cachedaccessorwrapper(
    name: str, accessor: C_prop[S, R], *, single_create: Literal[True]
) -> _CachedAccessorSingle[S, R]: ...


@overload
def _cachedaccessorwrapper(
    name: str, accessor: C_prop[S, R], *, single_create: Literal[False] = ...
) -> CachedProperty[S, R]: ...


@overload
def _cachedaccessorwrapper(
    name: str, accessor: C_prop[S, R], *, single_create: bool
) -> _CachedAccessorSingle[S, R] | CachedProperty[S, R]: ...


def _cachedaccessorwrapper(
    name: str, accessor: C_prop[S, R], *, single_create: bool = False
) -> _CachedAccessorSingle[S, R] | CachedProperty[S, R]:
    if single_create:
        return _CachedAccessorSingle(name, accessor)
    return _cachedaccessorcleared(name, accessor)


class AccessorMixin:  # (Generic[S, R]):
    """Mixin for accessor."""

    @classmethod
    def _register_accessor(
        cls, name: str, accessor: C_prop[S, R], single_create: bool = False
    ) -> None:
        """Most general accessor"""
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type {cls!r} is "
                "overriding a preexisting attribute with the same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(
            cls,
            name,
            _cachedaccessorwrapper(name, accessor, single_create=single_create),
        )

    @classmethod
    def register_accessor(
        cls, name: str, accessor: C_prop[S, R], single_create: bool = False
    ) -> None:
        """
        Register a property `name` to `class` of type `accessor(self)`

        Examples
        --------
        >>> class parent(AccessorMixin):
        ...     pass
        >>> class hello(AccessorMixin):
        ...     def __init__(self, parent):
        ...         self._parent = parent
        ...
        ...     def there(self):
        ...         return f"{type(self._parent)}"

        >>> parent.register_accessor("hello", hello)
        >>> x = parent()
        >>> x.hello.there()
        "<class 'lnpy.extensions.parent'>"
        """
        cls._register_accessor(name, accessor, single_create=single_create)

    @classmethod
    def decorate_accessor(
        cls, name: str, single_create: bool = False
    ) -> Callable[[C_prop[S, R]], C_prop[S, R]]:
        """
        Register a property `name` to `class` of type `accessor(self)`.

        Examples
        --------
        >>> class parent(AccessorMixin):
        ...     pass

        >>> @parent.decorate_accessor("hello")
        ... class hello(AccessorMixin):
        ...     def __init__(self, parent):
        ...         self._parent = parent
        ...
        ...     def there(self):
        ...         return f"{type(self._parent)}"

        >>> x = parent()
        >>> x.hello.there()
        "<class 'lnpy.extensions.parent'>"
        """

        def decorator(accessor: C_prop[S, R]) -> C_prop[S, R]:
            cls.register_accessor(name, accessor, single_create=single_create)
            return accessor

        return decorator
