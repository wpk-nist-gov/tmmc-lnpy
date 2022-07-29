import pytest

import lnPi
import pandas as pd
import numpy as np


from pathlib import Path

path_data = Path(__file__).parent / '../examples/LJ_mix'


@pytest.fixture
def lnz_values():

    # compare to this
    df = (
        pd.read_csv(path_data / 'out.ljmix4_full.t080.v512.r1.lnpi_o.dat.const_mu.csv.gz')
        .assign(
            lnz_0=lambda x: x['beta'] * x['mu_0'],
            lnz_1=lambda x: x['beta'] * x['mu_1'],
            betaOmega=lambda x: -x['beta'] * x['pressure'] * x['volume']
        )
    )

    lnzs = df[['lnz_0','lnz_1']].values[:5]

    return lnzs



@pytest.fixture
def ref():

    path = path_data / 'ljmix4_full.t080.v512.r1.lnpi_o.dat.gz'
    temp = 0.8
    state_kws = {'temp': temp, 'beta' : 1.0 / temp, 'volume' : 512}
    lnz =  np.array([-2.5, -2.5])

    return lnPi.MaskedlnPiDelayed.from_table(path, state_kws=state_kws, lnz=lnz).zeromax().pad()


@pytest.fixture
def phase_creator(ref):
    return lnPi.segment.PhaseCreator(nmax=2, nmax_peak=4, ref=ref, merge_kws=dict(efac=0.8))


def get_test_table(o, ref):
    return o.xge.table(keys=['betaOmega','nvec','PE','dens','betaF','S','betaG','edge_distance'], ref=ref)

def test_collection(phase_creator, ref, lnz_values):
    with lnPi.set_options(tqdm_leave=True, joblib_use=True, joblib_len_build=20, tqdm_len_build=10, tqdm_bar='text'):
        o = lnPi.CollectionlnPi.from_builder(lnz_values[:], phase_creator.build_phases, unstack=False)


    other = get_test_table(o, ref).unstack('sample').to_dataframe().reset_index()

    test = pd.read_csv(path_data / 'data_0.csv')

    pd.testing.assert_frame_equal(other, test)
