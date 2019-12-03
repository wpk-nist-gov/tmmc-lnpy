"""
routines to find constant molfracs
"""

import numpy as np
from scipy import optimize

from .segment import get_default_PhaseCreator

def find_lnz_molfrac(phaseID,
                    target,
                    lnzA,
                    lnzB,
                    comp=0,
                    ref=None,
                    build_phases=None,
                    build_kws=None,
                    nphases_max=None,
                    full_output=False,
                    tol=1e-4,
                    **kwargs):
    """
    calculate lnz which provides lnpi.molfracs_phaseIDs[phaseID,comp]==target

    Parameters
    ----------
    ref : MaskedlnPi
        object to reweight
    phaseID : int
        phaseID of the target
    target : float
        target molfraction
    lnzA,lnzB : lnz arrays bracketing solution
        only one index can vary between lnzA and lnzB
    comp : int (Default 0)
        the component ID of target molfraction
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

    if build_phases is None:
        assert nphases_max is not None
        build_phases = get_default_PhaseCreator(nphases_max).build_phases
    if build_kws is None:
        build_kws = {}

    lnzA = np.array(lnzA, dtype=float)
    lnzB = np.array(lnzB, dtype=float)

    msk = lnzA != lnzB
    if msk.sum() != 1:
        raise ValueError('only one value can vary between lnzA and lnzB')

    lnz_idx = np.where(msk)[0][0]
    lnz_in = lnzA.copy()

    a, b = sorted([x[lnz_idx] for x in [lnzA, lnzB]])

    def f(x):
        lnz = lnz_in[:]
        lnz[lnz_idx] = x

        p = build_phases(ref=ref, lnz=lnz, **build_kws)

        if phaseID in p.index:
            mf = p.xgce.molfrac.sel(phase=phaseID, component=comp).values
        else:
            mf = np.nan

        f.lnpi = p
        return mf - target

        # if lnpi.nphase == 1:
        #     mf = lnpi.molfrac.sel(phase=0, component=comp).values
        # else:
        #     mf = lnpi.molfrac_phase.sel(phase=phaseID, component=comp).values
        # f.lnpi = lnpi
        # return mf - target

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)
    r.residual = f(xx)

    if np.abs(r.residual) > tol:
        raise RuntimeError('something went wrong with solve')
    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi
