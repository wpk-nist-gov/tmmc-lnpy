"""
define accessors

This is inspired by xarray accessors
"""

import warnings
from itertools import chain 
from .cached_decorators import gcached

class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""

def _CachedAccessorWrapper(name, accessor):
    """
    Wrap accessor in a cached property

    This creates an property in the parent class with name `name`
    from `accessor(self)`

    this only gets initialized when called
    """
    @gcached(key=name, prop=True)
    def _get_prop(self):
        return accessor(self)
    return _get_prop





################################################################################
# List access
# class _CallableListResults(object):
#     """
#     if items of collection accessor are callable, then 
#     """

#     def __init__(self, parent, items):
#         self.parent = parent
#         self.items = items
#         self._cache = {}

#     @gcached(prop=False)
#     def __call__(self, *args, **kwargs):
#         results = [x(*args, **kwargs) for x in self.items]
#         if hasattr(self.parent, 'wrap_list_results'):
#             results = self.parent.wrap_list_results(results)
#         return results


# class _ListAccessor(object):
#     def __init__(self, parent, items):
#         self.parent = parent
#         self.items = items
#         self._cache = {}

#     def __getattr__(self, attr):
#         if attr not in self._cache:
#             try:
#                 result = [getattr(x, attr) for x in self.items]
#                 if callable(result[0]):
#                     # create a callable wrapper
#                     result = _CallableListResults(self.parent, result)
#                 else:
#                     if hasattr(self.parent, 'wrap_list_results'):
#                         result = self.parent.wrap_list_results(result)
#                 self._cache[attr] = result
#             except:
#                 raise AttributeError(f'no attribute {attr} found')
#         return self._cache[attr]

#     def __getitem__(self, idx):
#         return self.items[idx]

#     def __dir__(self):
#         heritage = dir(super(self.__class__, self))
#         #hide = []
#         x = self.items[0]
#         show = [
#             k for k in chain(
#                 self.__dict__.keys(), dir(x)
#                 # self.__class__.__dict__.keys()
#                 # x.__dict__.keys(),
#                 # x.__class__.__dict__.keys()
#             )] #  if k not in hide]
#         return sorted(heritage + show)

# def _CachedListPropertyWrapper(name):
#     """
#     get top level property from items
#     """
#     @gcached(key=name, prop=True)
#     def _get_prop(self):
#         results = [getattr(x, name) for x in self]
#         if callable(results[0]):
#             result = _CallableListResults(self, result)
#         else:
#             if hasattr(self, 'wrap_list_results'):
#                 results = self.wrap_list_results(results)
#         return results

# def _CachedListAccessorWrapper(name):
#     """
#     Wrap List accessor in cached property
#     """
#     @gcached(key=name, prop=True)
#     def _get_prop(self):
#         return _ListAccessor(self, [getattr(x, name) for x in self])
#     return _get_prop


# Old method using generic funcitons
# this has the bennefit that it can be applied without
# having the Mixin class in the objects.
# but this forces you to have the mixin class in the object
# so can't add properties to anything....
# def _register_accessor(name, accessor, parent_class, accessor_wrapper=_CachedAccessorWrapper):
#     if hasattr(parent_class, name):
#         warnings.warn(
#             'registration of accessor %r under name %r for type %r is '
#             'overriding a preexisting attribute with the same name.' %
#             (accessor, name, parent_class),
#             AccessorRegistrationWarning,
#             stacklevel=2)
#     setattr(parent_class, name, accessor_wrapper(name, accessor))

# # useful register/decorators
# def register_accessor(name, accessor, parent_class):
#     """
#     register a property `name` to `parent_class` of type `accessor(self)`
#     where `self` is an instance of `parent_class`


#     Examples
#     --------

#     class parent(object):
#         ...

#     class hello(object):
#         def __init__(self, parent):
#             self._parent = parent

#         def there(self):
#             return 'hello there {}'.format(type(self._parent))


#     register_accessor('hello', hello, parent, CachedAccessorWrapper)

 #     >>> x = parent()

#     >>> x.hello.there()
#     'hello there parent'

#     """
#     return _register_accessor(name, accessor, parent_class, accessor_wrapper=_CachedAccessorWrapper)

# def _decorate_accessor(name, parent_class, accessor_wrapper=_CachedAccessorWrapper):
#     def decorator(accessor):
#         _register_accessor(name, accessor, parent_class, accessor_wrapper)
#         return accessor
#     return decorator


# def decorate_accessor(name, parent_class):
#     """
#     decorate a class to add an accessor of name `name` to parent_class

#     Examples
#     --------

#     class parent(object):
#         ...

#     @register_accessor_decorator('hello', parent)
#     class hello(object):
#         def __init__(self, parent):
#             self._parent = parent

#         def there(self):
#             return 'hello there {}'.format(type(self._parent))

#     >>> x = parent()

#     >>> x.hello.there()
#     'hello there parent'
#     """
#     return _decorate_accessor(name, parent_class, accessor_wrapper=_CachedAccessorWrapper)


# def _register_listaccessor(name, parent_class, accessor_wrapper):
#     if hasattr(parent_class, name):
#         warnings.warn(
#             'registration of name %r for type %r is '
#             'overriding a preexisting attribute with the same name.' % (name, parent_class),
#             AccessorRegistrationWarning, stacklevel=2)
#     setattr(parent_class, name, accessor_wrapper(name))



# def register_listproperty(name, parent_class):
#     """
#     create an accessor to property

#     This is to give top level access to stuff

#     Examples
#     --------
#     class Child(object):
#         def __init__(self, a):
#             self.a = a

#         def b(self, x):
#             return self.a, x

#     class Wrapper(object):
#         def __init__(self, items):
#             self.items = items

#         def __getitem__(self, idx):
#             # must have getitem for this to work
#             return self.items[idx]

#         def wrap_list_results(self, result):
#             #anython special to do with resultsgoes here
#             return results

#     register_listproperty('a', Wrapper)
#     register_listproperty('b', Wrapper)

#     >>> x = [Child(i) for i in range(3)]

#     >>> w = Wrapper(x)

#     >>> w.a
#     [1, 2, 3]
#     >>> w.b('a')
#     [(1,'a'), (2,'a'), (3,'a')]
#     """
#     return _register_listaccessor(name, parent_class, _CachedListPropertyWrapper)


# def register_listaccessor(name, parent_class):
#     """
#     creat an accessor to elements of parent class.

#     Examples
#     --------
#     class Prop(object):
#         def __init__(self, prop)
#             self.prop = prop


#     class Child(object):
#         def __init__(self, a, b):
#             self.a = Prop(a)

#     class Wrapper(object):
#         def __init__(self, items):
#             self.items = items

#         def __getitem__(self, idx):
#             # must have getitem for this to work
#             return self.items[idx]

#         def wrap_results(self, result):
#             #anython special to do with resultsgoes here
#             return results

#     register_listaccessor('a', Wrapper)


#     >>> x = [Child(i) for i in range(3)]

#     >>> w = Wrapper(x)

#     >>> w.a.prop
#     [1, 2, 3]
#     """

#     return _register_listaccessor(name, parent_class, _CachedListAccessorWrapper)


# def _decorate_listaccessor(name, parent_class, accessor_wrapper):
#     def decorator(accessor):
#         _register_listaccessor(name, parent_class, accessor_wrapper)
#         return accessor
#     return decorator


# def decorate_listproperty(name, parent_class):
#     return _decorate_listaccessor(name, parent_class, _CachedListPropertyWrapper)


# def decorate_listaccessor(name, parent_class):
#     return _decorate_listaccessor(name, parent_class, _CachedListAccessorWrapper)



