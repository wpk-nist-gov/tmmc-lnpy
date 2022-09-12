from . import segment
from .ensembles import xCanonical, xGrandCanonical
from .lnpicollection import lnPiCollection

# Legacy stuff
from .lnpidata import MaskedlnPiDelayed, lnPiArray, lnPiMasked
from .lnpienergy import (
    merge_regions,
    wFreeEnergy,
    wFreeEnergyCollection,
    wFreeEnergyPhases,
)
from .maskedlnpi_legacy import MaskedlnPiLegacy
from .options import OPTIONS, set_options
from .segment import (
    BuildPhases_dmu,
    BuildPhases_mu,
    PhaseCreator,
    Segmenter,
    peak_local_max_adaptive,
)
from .stability import Binodals, Spinodals
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
    "lnPiArray",
    "lnPiMasked",
    "MaskedlnPiDelayed",
    "MaskedlnPiLegacy",
    "lnPiCollection",
    "segment",
    "xGrandCanonical",
    "xCanonical",
    "wFreeEnergyPhases",
    "wFreeEnergyCollection",
    "wFreeEnergy",
    "merge_regions",
    "Segmenter",
    "PhaseCreator",
    "peak_local_max_adaptive",
    "BuildPhases_mu",
    "BuildPhases_dmu",
    "dim_to_suffix",
    "set_options",
    "OPTIONS",
    "Spinodals",
    "Binodals",
]
