"""Publicly accessible classes/routines."""

# updated versioning scheme
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import combine, ensembles, lnpienergy, segment
    from .lnpidata import lnPiMasked
    from .lnpiseries import lnPiCollection
    from .options import OPTIONS, set_options
    from .segment import PhaseCreator
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=["ensembles", "lnpienergy", "segment", "combine"],
        submod_attrs={
            "lnpidata": ["lnPiMasked"],
            "lnpiseries": ["lnPiCollection"],
            "options": ["OPTIONS", "set_options"],
            "segment": ["PhaseCreator"],
        },
    )


try:
    __version__ = _version("tmmc-lnpy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "OPTIONS",
    "PhaseCreator",
    "__author__",
    "__email__",
    "__version__",
    "combine",
    "ensembles",
    "lnPiCollection",
    "lnPiMasked",
    "lnpienergy",
    "segment",
    "set_options",
]
