from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lnpy
import lnpy.stability

path_data = Path(__file__).parent / "../examples/HS_mix"


# function to tag 'LD' and 'HD' phases
def tag_phases2(x):
    if len(x) > 2:
        raise ValueError("bad tag function")
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


@pytest.fixture
def ref():
    return (
        lnpy.lnPiMasked.from_table(
            path_data / "nahs_asym_mix.07_07_07.r1.lnpi_o.dat",
            lnz=np.array([0.5, 0.5]),
            state_kws={"beta": 1.0, "volume": 1.0},
        )
        .zeromax()
        .pad()
    )


@pytest.fixture
def phase_creator(ref):
    return lnpy.segment.PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=ref,
        tag_phases=tag_phases2,
        merge_kws=dict(efac=0.8),
    )


import lnpy.examples


@pytest.fixture(params=[0, 1])
def obj(request, ref, phase_creator):
    if request.param == 0:
        return lnpy.examples.Example(
            ref=ref, phase_creator=phase_creator, build_phases=None
        )
    else:
        return lnpy.examples.hsmix_example()


@pytest.fixture
def lnz2():
    return 0.5


@pytest.fixture
def lnzs():
    return np.linspace(-8, 8, 20)


@pytest.fixture
def build_phases(obj, lnz2):
    return obj.phase_creator.build_phases_mu([None, lnz2])


def get_test_table(o, ref):
    return (
        o.xge.table(
            keys=[
                "betaOmega",
                "nvec",
                "PE",
                "dens",
                "betaF",
                "S",
                "betaG",
                "edge_distance",
            ],
            ref=ref,
        )
        .to_dataframe()
        .reset_index()
    )  # .to_csv('data_0.csv', index=False)


def test_collection(obj, build_phases, lnzs):
    ref = obj.ref

    c = lnpy.lnPiCollection.from_builder(lnzs, build_phases)
    c.spinodal(2, build_phases, inplace=True, unstack=True)
    c.binodal(2, build_phases, inplace=True, unstack=True)

    for path, obj in [
        ("data_0", c),
        ("data_0_spin", c.spinodal.access),
        ("data_0_bino", c.binodal.access),
    ]:
        test = pd.read_csv(path_data / (path + ".csv"))
        other = get_test_table(obj, ref)

        pd.testing.assert_frame_equal(test, other)

        test = pd.read_csv(path_data / (path + "_dw.csv"))
        other = obj.wfe.dw.to_frame().reset_index()

        pd.testing.assert_frame_equal(test, other)
