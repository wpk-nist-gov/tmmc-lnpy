"""
routines to find constant molfracs
"""


import numpy as np
from scipy import optimize

def find_mu_molfrac(ref,phaseID,molfrac,muA,muB,
                    comp=0,
                    reweight_kwargs={},
                    argmax_kwargs={},
                    phases_kwargs={},
                    ftag_phases=None,
                    full_output=False,
                    tol=1e-4,
                    **kwargs):
    """
    calculate mu which provides lnpi.molfracs_phaseIDs[phaseID]==molfrac
    """


    muA = np.array(muA,dtype=float)
    muB = np.array(muB,dtype=float)

    msk = muA != muB
    if msk.sum()!=1:
        raise ValueError('only one value can vary between muA and muB')        


    mu_idx = np.where(msk)[0][0]
    mu_in = muA.copy()

    

    a,b = sorted([x[mu_idx] for x in [muA,muB]])

    reweight_kwargs = dict(dict(ZeroMax=True),**reweight_kwargs)


    def f(x):
        mu = mu_in[:]
        mu[mu_idx] = x

        lnpi = ref.reweight(mu,**reweight_kwargs).to_phases(
            argmax_kwargs,phases_kwargs,ftag_phases)

        if lnpi.nphase==1:
            mf=lnpi.molfracs[0,comp]
        else:
            mf=lnpi.molfracs_phaseIDs[phaseID,comp]

        f.lnpi = lnpi

        return mf - molfrac



    xx,r = optimize.brentq(f,a,b,full_output=True,**kwargs)

    r.residual = f(xx)

    if np.abs(r.residual)>tol:
        raise RuntimeError('something went wrong with solve')
    
    if full_output:
        return f.lnpi,r
    else:
        return f.lnpi
