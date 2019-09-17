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


class _CallableListResultsCache(object):
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


class _CallableListResultsNoCache(object):
    """
    if items of collection accessor are callable, then 
    """

    def __init__(self, parent, items):
        self.parent = parent
        self.items = items
        self._cache = {}

    def __call__(self, *args, **kwargs):
        # get value
        results = [x(*args, **kwargs) for x in self.items]
        if hasattr(self.parent, 'wrap_list_results'):
            results = self.parent.wrap_list_results(results)
        return results

def _CallableListResults(parent, items, use_cache=False):
    if use_cache:
        cls = _CallableListResultsCache
    else:
        cls = _CallableListResultsNoCache

    return cls(parent=parent, items=items)


class _ListAccessor(object):
    def __init__(self, parent, items, cache_list=None):
        self.parent = parent
        self.items = items

        if cache_list is None:
            cache_list = []
        self._cache_list = cache_list

        self._cache = {}

    def __getattr__(self, attr):

        if attr in self._cache:
            # print('using cache')
            return self._cache[attr]

        try:
            result = [getattr(x, attr) for x in self.items]
            use_cache = attr in self._cache_list

            if callable(result[0]):
                # create a callable wrapper
                # always save this
                result = _CallableListResults(self.parent, result, use_cache=use_cache)
                # always cache this?
                # self._cache[attr] = result
            else:
                if hasattr(self.parent, 'wrap_list_results'):
                    result = self.parent.wrap_list_results(result)
                # print('caching')
                # self._cache[attr] = result
        except:
            raise AttributeError(f'no attribute {attr} found')

        # do caching?
        if use_cache:
            # print('caching')
            self._cache[attr] = result
        return result

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

def _CachedListPropertyWrapper(name, cache_list=None):
    """
    get top level property from items
    """

    if cache_list is not None and name in cache_list:
        cache = True
    else:
        cache = False


    #@gcached(key=name, prop=True)
    def _get_prop(self):
        results = [getattr(x, name) for x in self]
        if callable(results[0]):
            result = _CallableListResults(self, result)
        else:
            if hasattr(self, 'wrap_list_results'):
                results = self.wrap_list_results(results)
        return results
    if cache:
        _get_prop = gcached(key=name, prop=True)(_get_prop)
    else:
        _get_prop = property(_get_prop)

    return _get_prop

def _CachedListAccessorWrapper(name, cache_list=None):
    """
    Wrap List accessor in cached property
    """
    @gcached(key=name, prop=True)
    def _get_prop(self):
        return _ListAccessor(self, [getattr(x, name) for x in self], cache_list=cache_list)
    return _get_prop


class ListAccessorMixin(object):
    @classmethod
    def _register_listaccessor(cls, names, accessor_wrapper, **kwargs):
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
            setattr(cls, name, accessor_wrapper(name, **kwargs))

    @classmethod
    def register_listaccessor(cls, names, cache_list=None):
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
        return cls._register_listaccessor(names, accessor_wrapper=_CachedListAccessorWrapper, cache_list=cache_list)

    @classmethod
    def register_listproperty(cls, names, cache_list=None):
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
        return cls._register_listaccessor(names, accessor_wrapper=_CachedListPropertyWrapper, cache_list=cache_list)




def _decorate_listaccessor(names, accessor_wrapper, **kwargs):
    if isinstance(names, str):
        names = [names]
    def decorator(cls):
        for name in names:
            if hasattr(cls, name):
                warnings.warn(
                    'registration of name %r for type %r is '
                    'overriding a preexisting attribute with the same name.' % (name,cls),
                    AccessorRegistrationWarning, stacklevel=2)
            setattr(cls, name, accessor_wrapper(name, **kwargs))
        return cls
    return decorator


def decorate_listaccessor(names, cache_list=None):
    return _decorate_listaccessor(names, _CachedListAccessorWrapper, cache_list=cache_list)

def decorate_listproperty(names, cache_list=None):
    return _decorate_listaccessor(names, _CachedListPropertyWrapper, cache_list=cache_list)



