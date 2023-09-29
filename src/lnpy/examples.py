"""
Examples (:mod:`~lnpy.examples`)
================================
"""

from __future__ import annotations

try:
    import importlib_resources as resources
except ImportError:
    import importlib.resources as resources  # type: ignore[no-redef]

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import xarray as xr

# from .lnpiseries import lnPiCollection
from .segment import BuildPhasesBase, PhaseCreator
from .utils import dataset_to_lnpimasked

if TYPE_CHECKING:
    from typing import Any, Iterator, Literal, Sequence

    from ._typing import MyNDArray
    from .lnpidata import lnPiMasked

    _ExampleNames = Literal["lj_sub", "lj_sup", "ljmix_sup", "hsmix", "watermof"]


def json_to_dict(basename: str) -> dict[str, Any]:
    """
    Load a json file into a dict.

    All files names are relative to 'lnpy/data/'

    Parameters
    ----------
    basename : string
        Name of file to load in 'lnpy/data' directory.

    Returns
    -------
    output : dict

    """

    if basename.endswith(".gz"):
        import gzip

        fopen = gzip.open
    else:
        fopen = open  # type: ignore[assignment]

    with fopen(resources.files("lnpy.data").joinpath(basename), "r") as f:
        out = json.load(f)

    return out  # type: ignore


class ExampleDict(TypedDict):
    """Example dict"""

    lnPi_data: MyNDArray
    lnPi_mask: MyNDArray
    state_kws: dict[str, Any]
    extra_kws: dict[str, Any]
    lnz: MyNDArray


def load_example_dict(name: _ExampleNames) -> ExampleDict:
    """
    Load a dictionary of data

    Parameters
    ----------
    name : {'lj_sub', 'lj_sup', 'ljmix_sup', 'hsmix', 'watermof}
    """

    ref = load_example_lnpimasked(name)

    return ExampleDict(
        lnPi_data=ref.data,
        lnPi_mask=ref.mask,
        state_kws=ref.state_kws,
        extra_kws=ref.extra_kws,
        lnz=ref.lnz,
    )


def load_example_lnpimasked(name: _ExampleNames) -> lnPiMasked:
    """
    Load an example file

    Parameters
    ----------
    name : {'lj_sub', 'lj_sup', 'ljmix_sup', 'hsmix', 'watermof}
    """
    extensions = {
        "lj_sub": "json",
        "lj_sup": "json",
        "ljmix_sup": "json.gz",
        "hsmix": "json.gz",
        "watermof": "json",
    }

    assert name in extensions

    basename = f"{name}_example.{extensions[name]}"

    d = json_to_dict(basename)
    ds = xr.Dataset.from_dict(d)

    ref = dataset_to_lnpimasked(ds)
    return ref


@dataclass
class Example:
    """Dataclass to hold example data."""

    #: Reference state :class:`~lnpy.lnpidata.lnPiMasked`.
    ref: lnPiMasked
    #: :class:`~lnpy.segment.PhaseCreator` instance
    phase_creator: PhaseCreator
    #: Callable to build phases.
    build_phases: BuildPhasesBase | None

    def to_dict(self) -> dict[str, Any]:
        """Transform class to dictionary."""
        return asdict(self)

    def unpack(self, keys: list[str] | None = None) -> Iterator[Any]:
        """Unpack keys."""
        if keys is None:
            keys = ["ref", "phase_creator", "build_phases"]
        return (getattr(self, k) for k in keys)


def lj_sup_example() -> Example:
    """Create an :class:`Example` instance for a Lennard-Jones fluid (subcritical)"""
    ref = load_example_lnpimasked("lj_sup")

    phase_creator = PhaseCreator(
        nmax=1,
        nmax_peak=1,
        ref=ref,
        merge_kws={"efac": 0.8},
        tag_phases=None,
    )

    build_phases = phase_creator.build_phases_mu([None])

    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def tag_phases_single_comp_simple(x: Sequence[lnPiMasked]) -> MyNDArray:
    if len(x) > 2:
        raise ValueError("bad tag function")
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


def lj_sub_example() -> Example:
    """Create an :class:`Example` instance for a Lennard-Jones fluid (subcritical)"""
    ref = load_example_lnpimasked("lj_sub")

    phase_creator = PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=ref,
        merge_kws={"efac": 0.8},
        tag_phases=tag_phases_single_comp_simple,
    )

    build_phases = phase_creator.build_phases_mu([None])

    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def ljmix_sup_example() -> Example:
    """Create an :class:`Example` instance for a Lennard-Jones mixture (supercritical)."""
    ref = load_example_lnpimasked("ljmix_sup")

    phase_creator = PhaseCreator(
        nmax=1,
        ref=ref,
    )

    build_phases = phase_creator.build_phases_mu([None])
    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def hsmix_example() -> Example:
    """Create an :class:`Example` instance for a hard-sphere mixture."""

    def tag_phases(x: Sequence[lnPiMasked]) -> MyNDArray:
        if len(x) > 2:
            raise ValueError("bad tag function")
        argmax0 = np.array([xx.local_argmax()[0] for xx in x])
        return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)

    ref = load_example_lnpimasked("hsmix")

    phase_creator = PhaseCreator(
        nmax=2, nmax_peak=4, ref=ref, tag_phases=tag_phases, merge_kws={"efac": 0.8}
    )

    return Example(ref=ref, phase_creator=phase_creator, build_phases=None)


# def watermof_example():
#     def tag_phases(x):
#         if len(x) > 2:
#             raise ValueError("bad tag function")
#         argmax0 = np.array([xx.local_argmax()[0] for xx in x])
#         return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)

#     ref = load_example_lnpimasked('watermof')

#     phase_creator = PhaseCreator(
#         nmax=2,
#         nmax_peak=10,
#         merge_kws={'efac': 0.8}

#     )
