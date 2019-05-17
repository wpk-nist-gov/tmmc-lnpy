from functools import partial

from .core import MaskedlnPi, BaselnPiCollection, Phases, CollectionPhases
# from .accessors import (
#     register_accessor, decorate_accessor,
#     register_listaccessor, decorate_listaccessor,
#     register_listproperty, decorate_listproperty
# )


#core extensions
# class _Accessors(object):
#     def __init__(self, register_func, classes=None):

#         self.MaskedlnPi         = partial(register_func, parent_class=MaskedlnPi)
#         self.BaselnPiCollection = partial(register_func, parent_class=BaselnPiCollection)
#         self.Phases             = partial(register_func, parent_class=Phases)
#         self.CollectionPhases   = partial(register_func, parent_class=CollectionPhases)
#         # Note: don't use this because then autocomplete doesn't work for me
#         # if classes is None:
#         #     classes = [MaskedlnPi, BaselnPiCollection, Phases, CollectionPhases]
#         # for cls in classes:
#         #     name = cls.__name__
#         #     setattr(self, name, partial(register_func, parent_class=cls))


# # class _Decorate_and_register(object):
# #     def __init__(self, register_func, decorate_func, classes=None):
# #         self.register = _Accessors(register_func)
# #         self.decorate = _Accessors(decorate_func)


# # accessor = _Decorate_and_register(register_accessor, decorate_accessor)
# # listaccessor = _Decorate_and_register(register_listaccessor, decorate_listaccessor)
# # listproperty = _Decorate_and_register(register_listproperty, decorate_listproperty)



# class _AccessorsAll(object):
#     def __init__(self, accessor, listaccessor, listproperty):
#         self.accessor = _Accessors(accessor)
#         self.listaccessor = _Accessors(listaccessor)
#         self.listproperty = _Accessors(listproperty)


# register = _AccessorsAll(register_accessor, register_listaccessor, register_listproperty)
# decorate = _AccessorsAll(decorate_accessor, decorate_listaccessor, decorate_listproperty)







