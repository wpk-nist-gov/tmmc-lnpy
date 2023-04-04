from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lnpy

path_data = Path(__file__).parent / "../examples/LJ_mix"


@pytest.fixture
def ref():
    path = path_data / "ljmix4_full.t080.v512.r1.lnpi_o.dat.gz"
    temp = 0.8
    state_kws = {"temp": temp, "beta": 1.0 / temp, "volume": 512}
    lnz = np.array([-2.5, -2.5])

    return (
        lnpy.lnPiMasked.from_table(path, state_kws=state_kws, lnz=lnz).zeromax().pad()
    )


@pytest.fixture
def phase_creator(ref):
    return lnpy.segment.PhaseCreator(
        nmax=2, nmax_peak=4, ref=ref, merge_kws=dict(efac=0.8)
    )


import lnpy.examples


@pytest.fixture(params=[0, 1])
def obj(request, ref, phase_creator):
    if request.param == 0:
        return lnpy.examples.Example(
            ref=ref,
            phase_creator=phase_creator,
            build_phases=phase_creator.build_phases,
        )

    else:
        return lnpy.examples.ljmix_sup_example()


def get_test_table(o, ref):
    return o.xge.table(
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


def test_collection(obj):
    ref, phase_creator = obj.unpack(["ref", "phase_creator"])

    test = pd.read_csv(path_data / "data_0.csv")

    lnz_values = test[["lnz_0", "lnz_1"]].values

    with lnpy.set_options(
        tqdm_leave=True,
        joblib_use=True,
        joblib_len_build=20,
        tqdm_len_build=10,
        tqdm_bar="text",
    ):
        o = lnpy.lnPiCollection.from_builder(
            lnz_values[:], phase_creator.build_phases, unstack=False
        )

    other = get_test_table(o, ref).unstack("sample").to_dataframe().reset_index()

    pd.testing.assert_frame_equal(other, test)
