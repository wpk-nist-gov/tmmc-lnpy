"""
routines to find binodal point between two phases
"""

import numpy as np
from scipy import optimize


def get_binodal_point(ref,
                      IDs,
                      muA,
                      muB,
                      reweight_kwargs={},
                      full_output=False,
                      **kwargs):
    """
    calculate binodal point where Omega[ID[0]]==Omega[ID[1]]

    Parameters
    ----------
    ref : lnPi_phases object
        object to reweight

    IDs : (ID0,ID1)
        phaseIDs of pair to equate

    muA,muB : mu arrays bracketing solution

    reweight_kwargs : dict
        extra arguments to reweight
    
    full_output : bool (Default False)
        if True, return solve stats

    **kwargs : extra arguments to scipy.optimize.brentq

    Returns
    -------
    binodal : lnPi_phases object at binodal point

    stats : solve stats object from brentq (optional, returned if full_output is True)
     """

    assert len(IDs) == 2

    muA = np.asarray(muA)
    muB = np.asarray(muB)

    msk = muA != muB
    if msk.sum() != 1:
        raise ValueError('only one value can vary between muA and muB')

    mu_idx = np.where(msk)[0][0]
    mu_in = muA.copy()

    a, b = sorted([x[mu_idx] for x in [muA, muB]])

    reweight_kwargs = dict(dict(zeromax=True), **reweight_kwargs)

    def f(x):
        mu = mu_in[:]
        mu[mu_idx] = x
        c = ref.reweight(mu, **reweight_kwargs)
        f.lnpi = c

        Omegas = c.omega_phase()

        return Omegas[IDs[0]] - Omegas[IDs[1]]

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi
