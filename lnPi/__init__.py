from . import segment
from .collection import MaskedDataCollection
from .ensembles import xCanonical, xGrandCanonical
from .localenergy import (
    merge_regions,
    wFreeEnergy,
    wFreeEnergyCollection,
    wFreeEnergyPhases,
)

# Legacy stuff
from .maskeddata import MaskedData, MaskedlnPiDelayed, lnPiData
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
    "lnPiData",
    "MaskedData",
    "MaskedlnPiDelayed",
    "MaskedlnPiLegacy",
    "MaskedDataCollection",
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
