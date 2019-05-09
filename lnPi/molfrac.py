"""
routines to find constant molfracs
"""

import numpy as np
from scipy import optimize


def find_mu_molfrac(ref,
                    phaseID,
                    target,
                    muA,
                    muB,
                    comp=0,
                    reweight_kwargs={},
                    full_output=False,
                    tol=1e-4,
                    **kwargs):
    """
    calculate mu which provides lnpi.molfracs_phaseIDs[phaseID,comp]==target

    Parameters
    ----------
    ref : lnPi_phases object
        object to reweight

    phaseID : int
        phaseID of the target

    target : float
        target molfraction

    muA,muB : mu arrays bracketing solution
        only one index can vary between muA and muB

    comp : int (Default 0)
        the component ID of target molfraction

    reweight_kwargs : dict
        extra arguments to ref.reweight

    full_output : bool (Default False)
        if True, return solve stats

    tol : float (default 1e-4)
        solver tolerance

    **kwargs : extra arguments to scipy.optimize.brentq

    Returns
    --------
    output : lnPi_phases object
        object with desired molfraction

    info : solver info (optional, returned if full_output is `True`)

    """

    muA = np.array(muA, dtype=float)
    muB = np.array(muB, dtype=float)

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

        lnpi = ref.reweight(mu, **reweight_kwargs)

        if lnpi.nphase == 1:
            mf = lnpi.molfrac.sel(phase=0, component=comp).values
        else:
            mf = lnpi.molfrac_phase.sel(phase=phaseID, component=comp).values

        f.lnpi = lnpi

        return mf - target

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)

    if np.abs(r.residual) > tol:
        raise RuntimeError('something went wrong with solve')

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi
