from . import collectionlnpi, extensions, segment, wlnPi, xlnPi
from .collectionlnpi import CollectionlnPi
from .maskedlnpi import MaskedlnPi, MaskedlnPiDelayed, lnPiData
from .maskedlnpi_legacy import MaskedlnPiLegacy
from .options import OPTIONS, set_options
from .utils import dim_to_suffix

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("cmomy").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "lnPiData",
    "MaskedlnPi",
    "MaskedlnPiDelayed",
    "MaskedlnPiLegacy",
    "CollectionlnPi",
    "collectionlnpi",
    "segment",
    "extensions",
    "xlnPi",
    "wlnPi",
    "dim_to_suffix",
    "set_options",
    "OPTIONS",
]
