"""
Legacy interface to :mod:`lnpy` (:mod:`lnPi`)
=============================================
"""

from warnings import warn

from numpy import VisibleDeprecationWarning

from lnpy import __version__, options, segment, set_options
from lnpy.lnpidata import lnPiMasked as MaskedlnPiDelayed
from lnpy.lnpiseries import lnPiCollection as CollectionlnPi

from . import stability

msg = """\
lnPi is deprecated.  Please transition to using lnpy instead.  Name changes are:

Modules:
lnPi -> lnpy
lnPi.collectionlnpiutils -> lnpy.lnpicollectionutils

Classes:
MaskedlnPiDelayed -> lnPiMasked
CollectionlnPi    -> lnPiCollection
"""


warn(msg, VisibleDeprecationWarning)


__all__ = [
    "MaskedlnPiDelayed",
    "CollectionlnPi",
    "segment",
    "stability",
    "options",
    "set_options",
    "__version__",
]
