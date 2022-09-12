from .ensembles import xCanonical, xGrandCanonical
from .lnpidata import MaskedlnPiDelayed, lnPiArray, lnPiMasked
from .lnpienergy import (
    merge_regions,
    wFreeEnergy,
    wFreeEnergyCollection,
    wFreeEnergyPhases,
)
from .lnpiseries import lnPiCollection
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

# Legacy stuff


# updated versioning scheme
try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]


try:
    __version__ = _version("lnpy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "lnPiArray",
    "lnPiMasked",
    "lnPiCollection",
    "MaskedlnPiDelayed",
    "MaskedlnPiLegacy",
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
