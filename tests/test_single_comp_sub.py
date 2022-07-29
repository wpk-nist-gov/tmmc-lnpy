import pytest


import lnPi, lnPi.stability
import json
import pandas as pd
import numpy as np


from pathlib import Path


path_data = Path(__file__).parent / '../examples/LJ_cfs_2.5sig'

def get_lnz(path):
    kB = 1.3806503e-23  # J/K
    Na = 6.02214199e23  # 1/mol
    kg = 1.66054e-27  # 1/amu
    hPlanck = 6.62606876e-34  # Js

    with open(path) as f:
        metadata = json.load(f)

    deBroglie = (
        hPlanck
        / np.sqrt(
            2.0
            * np.pi
            * (metadata["mass"] * kg)
            * kB
            * (metadata["T*"] * metadata["eps_kB"])
        )
        * 1.0e10
    )  # ang

    mu = metadata["mu*"] + 3.0 * metadata["T*"] * np.log(metadata["sigma"] / deBroglie)
    temp = metadata["T*"]

    lnz = mu / temp

    return lnz, {"beta": 1.0 / temp, "volume": metadata["V*"]}


def tag_phases2(x):
    if len(x) > 2:
        raise ValueError('bad tag function')
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


# get meta data
@pytest.fixture
def ref():
    lnz, state_kws = get_lnz(path_data /"t072871.metadata.json")
    print(state_kws)

    # read in potential energy
    pe = pd.read_csv(
        path_data /"ljsf.t072871.bulk.v512.r1.energy.dat", header=None, sep="\s+", names=["n", "e"]
    )["e"].values

    return (
        lnPi.MaskedlnPiDelayed.from_table(
            path_data / "ljsf.t072871.bulk.v512.r1.lnpi.dat",
            fill_value=np.nan,
            lnz=lnz,
            # state_kws needs to be defined if want to calculate some properties
            # down the road
            state_kws=state_kws,
            # extra_kws is where you pass things which will be passed along to
            # new lnPis, like potential energy
            extra_kws={"PE": pe},
        )
        .zeromax()
        .pad()
    )


@pytest.fixture
def phase_creator(ref):
    return lnPi.segment.PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=ref,
        merge_kws=dict(efac=0.8),
        # if want to id phases based on some
        # callable, then set this.
        # the callable is passes a list of
        # MaskedlnPi objects, and should return
        # a list of phaseid
        # tag_phases=None
        tag_phases=tag_phases2
    )


@pytest.fixture
def build_phases(phase_creator):
    # choose a particular factory method for creating phases
    return phase_creator.build_phases_mu([None])


@pytest.fixture
def test_table():
    table = pd.read_csv(path_data / "data_1.csv")
    return table


def get_test_table(o, ref):
    return o.xge.table(keys=['betaOmega','nvec','PE','dens','betaF','S','betaG','edge_distance'], ref=ref).to_dataframe().reset_index()#.to_csv('data_0.csv', index=False)



def test_collection_properties(build_phases, test_table, ref):
    # for big builds, take advantage of progress bar, and parallel builds
    lnzs = np.linspace(-10, 3, 2000)

    # by default, progress bar hides itself after completion.  use context manager to keep it
    # note that for this example (where only have a single phase), doesn't really make a difference
    with lnPi.set_options(tqdm_leave=True, joblib_use=False, tqdm_bar='text'):
        o = lnPi.CollectionlnPi.from_builder(lnzs, build_phases)

    other = get_test_table(o, ref)

    pd.testing.assert_frame_equal(test_table, other)


@pytest.fixture
def test_table_can():
    table = pd.read_csv(path_data / "data_1_can.csv")

    return table


def get_test_table_can(ref):
    return ref.xce.table(keys=['S','Z','betaE','betaF','betaOmega','betamu','dens','ntot']).to_dataframe().reset_index()

def test_canonical_properties(ref, test_table_can):
    other = get_test_table_can(ref)
    pd.testing.assert_frame_equal(other, test_table_can)



def test_nice_grid(build_phases, ref):
    import lnPi.collectionlnpiutils

    with lnPi.set_options(joblib_use=True):
        o_course, o = lnPi.collectionlnpiutils.limited_collection(build_phases, dlnz=0.01, offsets=[-10, +10],
                                                                  even_grid=True, # but lnzs on same grid as dlnz
                                                                  digits=2, #round lnzs to this number of digits
                                                                  edge_distance_min=10, dens_min=0.001)

    other_course = get_test_table(o_course, ref)
    test_course = pd.read_csv(path_data / 'data_1_course.csv')
    pd.testing.assert_frame_equal(other_course, test_course)


    other_fine = get_test_table(o, ref)
    test = pd.read_csv(path_data / 'data_1_fine.csv')
    pd.testing.assert_frame_equal(other_fine, test)


    # spinodal/binodal
    o_course.spinodal(2, build_phases)
    o_course.binodal(2, build_phases)


    other = get_test_table(o_course.spinodal.access, ref)
    test = pd.read_csv(path_data / 'data_1_spin.csv')
    pd.testing.assert_frame_equal(other, test)


    other = get_test_table(o_course.binodal.access, ref)
    test = pd.read_csv(path_data / 'data_1_bino.csv')
    pd.testing.assert_frame_equal(other, test)


    # check dw
    other = o_course.spinodal.access.wlnPi.dw.to_frame().reset_index()
    test = pd.read_csv(path_data / 'data_1_spin_dw.csv')
    pd.testing.assert_frame_equal(other, test)
