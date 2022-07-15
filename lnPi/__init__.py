from .maskedlnpi import MaskedlnPi
from .maskedlnpidelayed import MaskedlnPiDelayed
from .collectionlnpi import CollectionlnPi
from . import collectionlnpi
from . import segment
from . import extensions
from . import xlnPi


from .utils import dim_to_suffix
from .options import set_options, OPTIONS



try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("cmomy").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"
