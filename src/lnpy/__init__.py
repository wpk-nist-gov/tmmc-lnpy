"""Publicly accessible classes/routines."""
from . import ensembles, lnpienergy, segment
from .lnpidata import lnPiMasked
from .lnpiseries import lnPiCollection
from .options import OPTIONS, set_options
from .segment import PhaseCreator

# updated versioning scheme
try:
    from ._version import __version__
except Exception:
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "lnPiMasked",
    "lnPiCollection",
    "segment",
    "ensembles",
    "lnpienergy",
    "PhaseCreator",
    "set_options",
    "OPTIONS",
]
