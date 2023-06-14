"""Publicly accessible classes/routines."""
# from .ensembles import xCanonical, xGrandCanonical
# from .lnpidata import MaskedlnPiDelayed, lnPiArray, lnPiMasked
from . import ensembles, lnpienergy, segment
from .lnpidata import lnPiMasked

# from .lnpienergy import (
#     merge_regions,
#     wFreeEnergy,
#     wFreeEnergyCollection,
#     wFreeEnergyPhases,
# )
from .lnpiseries import lnPiCollection

# from .maskedlnpi_legacy import MaskedlnPiLegacy
from .options import OPTIONS, set_options
from .segment import PhaseCreator

# from .stability import Binodals, Spinodals
# from .utils import dim_to_suffix

# Legacy stuff


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
    # "lnPiArray",
    "lnPiMasked",
    "lnPiCollection",
    "segment",
    "ensembles",
    "lnpienergy",
    # "MaskedlnPiDelayed",
    # "MaskedlnPiLegacy",
    # "xGrandCanonical",
    # "xCanonical",
    # "wFreeEnergyPhases",
    # "wFreeEnergyCollection",
    # "wFreeEnergy",
    # "merge_regions",
    # "Segmenter",
    "PhaseCreator",
    # "peak_local_max_adaptive",
    # "BuildPhases_mu",
    # "BuildPhases_dmu",
    # "dim_to_suffix",
    "set_options",
    "OPTIONS",
    # "Spinodals",
    # "Binodals",
]
