"""
Routines to load/run examples
"""

try:
    import importlib_resources as resources
except ImportError:
    import importlib.resources as resources

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

# from .lnpiseries import lnPiCollection
from .lnpidata import lnPiMasked
from .segment import PhaseCreator
from .utils import dataset_to_lnpimasked


def json_to_dict(basename):
    """Load a json file into a dict.

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
        fopen = open

    with fopen(resources.files("lnpy.data").joinpath(basename), "r") as f:
        out = json.load(f)

    return out


def load_example_dict(name):
    """
    Load a dictionary of data

    Parameters
    ----------
    name : {'lj_sub', 'lj_sup', 'ljmix_sup', 'hsmix', 'watermof}
    """

    ref = load_example_maskddata(name)

    return {
        "lnPi_data": ref.data,
        "lnPi_mask": ref.mask,
        "state_kws": ref.state_kws,
        "extra_kws": ref.extra_kws,
        "lnz": ref.lnz,
    }


def load_example_maskddata(name):
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
    ref: lnPiMasked
    phase_creator: PhaseCreator
    build_phases: Callable

    def to_dict(self):
        return asdict(self)

    def unpack(self, keys=None):
        if keys is None:
            keys = ["ref", "phase_creator", "build_phases"]
        return (getattr(self, k) for k in keys)


def lj_sup_example():
    ref = load_example_maskddata("lj_sup")

    phase_creator = PhaseCreator(
        nmax=1,
        nmax_peak=1,
        ref=ref,
        merge_kws={"efac": 0.8},
        tag_phases=None,
    )

    build_phases = phase_creator.build_phases_mu([None])

    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def tag_phases_single_comp_simple(x):
    if len(x) > 2:
        raise ValueError("bad tag function")
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


def lj_sub_example():

    ref = load_example_maskddata("lj_sub")

    phase_creator = PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=ref,
        merge_kws={"efac": 0.8},
        tag_phases=tag_phases_single_comp_simple,
    )

    build_phases = phase_creator.build_phases_mu([None])

    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def ljmix_sup_example():
    ref = load_example_maskddata("ljmix_sup")

    phase_creator = PhaseCreator(
        nmax=1,
        ref=ref,
    )

    build_phases = phase_creator.build_phases
    return Example(ref=ref, phase_creator=phase_creator, build_phases=build_phases)


def hsmix_example():
    def tag_phases(x):
        if len(x) > 2:
            raise ValueError("bad tag function")
        argmax0 = np.array([xx.local_argmax()[0] for xx in x])
        return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)

    ref = load_example_maskddata("hsmix")

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

#     ref = load_example_maskddata('watermof')

#     phase_creator = PhaseCreator(
#         nmax=2,
#         nmax_peak=10,
#         merge_kws={'efac': 0.8}

#     )
