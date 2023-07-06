"""Publicly accessible classes/routines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ensembles, lnpienergy, segment  # noqa: TCH004
    from .lnpidata import lnPiMasked  # noqa: TCH004
    from .lnpiseries import lnPiCollection  # noqa: TCH004
    from .options import OPTIONS, set_options  # noqa: TCH004
    from .segment import PhaseCreator  # noqa: TCH004
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=["ensembles", "lnpienergy", "segment"],
        submod_attrs={
            "lnpidata": ["lnPiMasked"],
            "lnpiseries": ["lnPiCollection"],
            "options": ["OPTIONS", "set_options"],
            "segment": ["PhaseCreator"],
        },
    )


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
