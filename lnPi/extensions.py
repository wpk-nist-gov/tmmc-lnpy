"""
define accessors

This is inspired by xarray accessors
"""

import warnings
from itertools import chain 
from .cached_decorators import gcached

class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""



class _CachedAccessorSingle(object):
    """
    Custom property-like object (descriptor).
    this gets created once on call
    """

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        try:
            accessor_obj = self._accessor(obj)
        except AttributeError:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            raise RuntimeError('error initializing %r accessor.' % self._name)
        # Replace the property with the accessor object. Inspired by:
        # http://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # AttrAccessMixin.
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj

def _CachedAccessorCleared(name, accessor):
    """
    Wrap accessor in a cached property

    This creates an property in the parent class with name `name`
    from `accessor(self)`

    this only gets initialized when called

    NOTE : it gets deleted if self._cache is cleared.

    If you only want to create it once, then use the Single access wrapper below
    """
    @gcached(key=name, prop=True)
    def _get_prop(self):
        return accessor(self)
    return _get_prop


def _CachedAccessorWrapper(name, accessor, single_create=False):
    if single_create:
        return _CachedAccessorSingle(name, accessor)
    else:
        return _CachedAccessorCleared(name, accessor)



class AccessorMixin(object):
    @classmethod
    def _register_accessor(cls, name, accessor, single_create=False):
        """
        most general accessor
        """
        if hasattr(cls, name):
            warnings.warn(
                'registration of accessor %r under name %r for type %r is '
                'overriding a preexisting attribute with the same name.' %
                (accessor, name, cls),
                AccessorRegistrationWarning, stacklevel=2)
        setattr(cls, name, _CachedAccessorWrapper(name, accessor, single_create))

    @classmethod
    def register_accessor(cls, name, accessor, single_create=False):
        """
        register a property `name` to `class` of type `accessor(self)`
        Examples
        --------
        class parent(AccessorMixin):
            pass

        class hello(AccessorMixin):
            def __init__(self, parent):
                self._parent = parent
            def there(self):
                return 'hello there {}'.format(type(self._parent))
        parent.register_accessor('hello', hello)
        >>> x = parent()
        >>> x.hello.there()
        'hello there parent'
        """
        return cls._register_accessor(name, accessor, single_create)

    @classmethod
    def decorate_accessor(cls, name, single_create=False):
        """
        register a property `name` to `class` of type `accessor(self)`
        Examples
        --------
        class parent(AccessorMixin):
            pass
        @parent.decorate('hello)
        class hello(AccessorMixin):
            def __init__(self, parent):
                self._parent = parent
            def there(self):
                return 'hello there {}'.format(type(self._parent))
        >>> x = parent()
        >>> x.hello.there()
        'hello there parent'
        """
        def decorator(accessor):
            cls.register_accessor(name, accessor, single_create)
            return accessor
        return decorator



################################################################################
# List access

def deep_cache(func):
    func.__dict__['_deep_cache'] = True
    return func


class _CallableListResults(object):
    """
    if items of collection accessor are callable, then 
    """

    def __init__(self, parent, items):
        self.parent = parent
        self.items = items
        self._cache = {}
    @gcached(prop=False)
    def __call__(self, *args, **kwargs):
        # get value
        results = [x(*args, **kwargs) for x in self.items]
        if hasattr(self.parent, 'wrap_list_results'):
            results = self.parent.wrap_list_results(results)
        return results


class _ListAccessor(object):
    def __init__(self, parent, items):
        self.parent = parent
        self.items = items
        self._cache = {}

    def __getattr__(self, attr):
        if attr not in self._cache:
            try:
                result = [getattr(x, attr) for x in self.items]
                if callable(result[0]):
                    # create a callable wrapper
                    result = _CallableListResults(self.parent, result)
                else:
                    if hasattr(self.parent, 'wrap_list_results'):
                        result = self.parent.wrap_list_results(result)
                self._cache[attr] = result
            except:
                raise AttributeError(f'no attribute {attr} found')
        return self._cache[attr]

    def __getitem__(self, idx):
        return self.items[idx]

    def __dir__(self):
        heritage = dir(super(self.__class__, self))
        #hide = []
        x = self.items[0]
        show = [
            k for k in chain(
                self.__dict__.keys(), dir(x)
                # self.__class__.__dict__.keys()
                # x.__dict__.keys(),
                # x.__class__.__dict__.keys()
            )] #  if k not in hide]
        return sorted(heritage + show)

def _CachedListPropertyWrapper(name):
    """
    get top level property from items
    """
    @gcached(key=name, prop=True)
    def _get_prop(self):
        results = [getattr(x, name) for x in self]
        if callable(results[0]):
            result = _CallableListResults(self, result)
        else:
            if hasattr(self, 'wrap_list_results'):
                results = self.wrap_list_results(results)
        return results
    return _get_prop

def _CachedListAccessorWrapper(name):
    """
    Wrap List accessor in cached property
    """
    @gcached(key=name, prop=True)
    def _get_prop(self):
        return _ListAccessor(self, [getattr(x, name) for x in self])
    return _get_prop


class ListAccessorMixin(object):
    @classmethod
    def _register_listaccessor(cls, names, accessor_wrapper):
        """
        most general accessor
        """
        if isinstance(names, str):
            names = [names]
        for name in names:
            if hasattr(cls, name):
                warnings.warn(
                    'registration of name %r for type %r is '
                    'overriding a preexisting attribute with the same name.' % (name, cls),
                    AccessorRegistrationWarning, stacklevel=2)
            setattr(cls, name, accessor_wrapper(name))

    @classmethod
    def register_listaccessor(cls, names):
        """
        create an accessor to property

        This is to give top level access to stuff

        Examples
        --------
        class Child(object):
            def __init__(self, a):
                self.a = a
            def b(self, x):
                return self.a, x
        class Wrapper(ListAccessorMixin):
            def __init__(self, items):
                self.items = items

            def __getitem__(self, idx):
                # must have getitem for this to work
                return self.items[idx]

            def wrap_list_results(self, result):
                #anython special to do with resultsgoes here
                return results

        Wrapper.register_listproperty(['a', 'b'])
        >>> x = [Child(i) for i in range(3)]
        >>> w = Wrapper(x)
        >>> w.a
        [1, 2, 3]
        >>> w.b('a')
        [(1,'a'), (2,'a'), (3,'a')]
        """
        return cls._register_listaccessor(names, accessor_wrapper=_CachedListAccessorWrapper)

    @classmethod
    def register_listproperty(cls, names):
        """
        creat an accessor to elements of parent class.

        Examples
        --------
        class Prop(object):
            def __init__(self, prop)
                self.prop = prop
        class Child(object):
            def __init__(self, a, b):
                self.a = Prop(a)
        class Wrapper(ListAccessorMixin):
            def __init__(self, items):
                self.items = items
            def __getitem__(self, idx):
                # must have getitem for this to work
                return self.items[idx]
            def wrap_list_results(self, result):
                #anython special to do with resultsgoes here
                return results

        wrapper.register_listaccessor('a')
        >>> x = [Child(i) for i in range(3)]
        >>> w = Wrapper(x)
        >>> w.a.prop
        [1, 2, 3]
        """
        return cls._register_listaccessor(names, accessor_wrapper=_CachedListPropertyWrapper)




def _decorate_listaccessor(names, accessor_wrapper):
    if isinstance(names, str):
        names = [names]
    def decorator(cls):
        for name in names:
            if hasattr(cls, name):
                warnings.warn(
                    'registration of name %r for type %r is '
                    'overriding a preexisting attribute with the same name.' % (name,cls),
                    AccessorRegistrationWarning, stacklevel=2)
            setattr(cls, name, accessor_wrapper(name))
        return cls
    return decorator


def decorate_listaccessor(names):
    return _decorate_listaccessor(names, _CachedListAccessorWrapper)

def decorate_listproperty(names):
    return _decorate_listaccessor(names, _CachedListPropertyWrapper)



