from . import collectionlnpi, extensions, segment, xlnPi
from .collectionlnpi import CollectionlnPi
from .maskedlnpi import MaskedlnPi
from .maskedlnpidelayed import MaskedlnPiDelayed
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
    "MaskedlnPi",
    "MaskedlnPiDelayed",
    "CollectionlnPi",
    "collectionlnpi",
    "segment",
    "extensions",
    "xlnPi",
    "dim_to_suffix",
    "set_options",
    "OPTIONS",
]
