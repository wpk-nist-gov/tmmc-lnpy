import numpy as np

from scipy import optimize


def _initial_bracket_spinodal_right(C,
                                    ID,
                                    mu_in,
                                    efac=1.0,
                                    dmu=0.5,
                                    vmax=1e20,
                                    ntry=20,
                                    step=+1,
                                    reweight_kwargs={},
                                    DeltabetaE_kwargs={}):
    """
    find initial bracketing lnpi_phases of phaseID==ID bracketing point where DeltabetaE_phaseIDS()[ID]==efac

    Parameters
    ----------
    C : lnPi_collection
        initial estimates to work from

    ID : int
        phaseID to work with

    mu_in : list
        list with value of static chem pot, and None for variable. e.g.,
        mu_in=[None,0.0] implies mu[0] is variable, and mu[1]=0.0

    efac : float (Default 1.0)
        cutoff value for spinodal

    dmu : float (Default 0.5)
        factor to kick back if C doesn't already have left and right bounds

    vmax : float (default 1e20)
        value indicating no transition, but phaseID==ID present

    ntry : int (Default 20)
        number of times to try kicking forward/backwards to find bracket

    step : int (Default +1)
        if +1, step forward
        if -1, step backward 
    
    reweight_kwargs : dict
        extra arguments to reweight

    DeltabetaE_kwargs : dict
        extra arguemtns to lnPi.DeltabetaE_phaseIDs

    Returns
    -------
    left,right: lnpi_phases
        left and right bracketing lnPi_phases objects
    
    """

    #use lnpi_phase reference
    ref = C[0]

    reweight_kwargs = dict(dict(ZeroMax=True), **reweight_kwargs)

    #mu which varies
    mu_idx = mu_in.index(None)

    #delta E
    dE = C.DeltabetaE_phaseIDs(**DeltabetaE_kwargs)[:, ID]
    Omegas = C.Omega_phase().sel(phase=ID).values

    #find locations where have 'ID'
    msk = C.has_phaseIDs[:, ID]
    if msk.sum() == 0:
        raise ValueError('no phaseID %i' % ID)

    w = np.where(msk)[0]

    #left
    left = None
    for i in w[-1::-1]:
        if dE[i] > efac and np.isfinite(Omegas[i]):
            left = C[i]
            break

    if left is None:
        #need to find a new value mu bounding thing
        new_mu = mu_in[:]
        new_mu[mu_idx] = C[w[0]].mu[mu_idx]
        new_mu = np.asarray(new_mu)

        for i in range(ntry):
            new_mu[mu_idx] -= step * dmu

            t = ref.reweight(new_mu, **reweight_kwargs)

            if t.DeltabetaE_phaseIDs(**DeltabetaE_kwargs)[ID]>efac and \
               np.isfinite(t.Omega_phase().sel(phase=ID).values):
                left = t
                break
        if left is None:
            raise RuntimeError('could not find left')

    #right
    #first, is there a value in C where has_phase and dE<efac?
    right = None
    for i in w[-1::-1]:
        if dE[i] < efac:
            right = C[i]
            break

    if right is None:
        if w[-1] + 1 < len(C):
            right = C[w[-1] + 1]
        else:
            new_mu = mu_in[:]
            new_mu[mu_idx] = C[w[-1]].mu[mu_idx]
            new_mu = np.asarray(new_mu)

            for i in range(ntry):
                new_mu[mu_idx] += step * dmu

                t = ref.reweight(new_mu, **reweight_kwargs)

                if not t.has_phaseIDs[ID]:
                    right = t
                    break
            if right is None:
                raise RuntimeError('could not find right')

    return left, right


def _refine_bracket_spinodal_right(L,
                                   R,
                                   ID,
                                   efac=1.0,
                                   nmax=30,
                                   vmax=1e20,
                                   vmin=0.0,
                                   reweight_kwargs={},
                                   DeltabetaE_kwargs={},
                                   close_kwargs={}):
    """
    find refined bracket with efac<DeltabetaE_left<vmax and vmin<DeltabetaE_right<efac

    Parameters
    ----------
    L,R : lnpi_phases objects
        left and right initial brackets

    ID : int
        phaseID to work with

    efac : float (Default 1.0)
        cutoff value for spinodal

    nmax : int (Default 30)
        max number of interations

    vmin,vmax : see above

    reweight_kwargs : dict
        extra arguments to reweight

    DeltabetaE_kwargs : dict
        extra arguments to lnPi_phases.DeltabetaE_phaseIDs

    close_kwargs : dict
        arguments to np.allclose

    Returns
    -------
    left,right : lnpi_phases objects
        left and right phases bracketing spinodal
    
    r : scipy.optimize.zeros.RootResults object
    """

    ref = L
    reweight_kwargs = dict(dict(ZeroMax=True), **reweight_kwargs)

    doneLeft = False
    doneRight = False

    left = L
    right = R

    close_kwargs = dict(dict(atol=1e-5), **close_kwargs)

    for i in range(nmax):
        v = left.DeltabetaE_phaseIDs()[ID]
        if v < vmax and v > efac:
            doneLeft = True

        v = right.DeltabetaE_phaseIDs(**DeltabetaE_kwargs)[ID]
        if v > vmin and v < efac:
            doneRight = True

        #########
        #checks
        if doneLeft and doneRight:
            #find bracket
            r = optimize.zeros.RootResults(
                root=None, iterations=i, function_calls=i, flag=1)
            return left, right, r

        ########
        #converged?
        if np.allclose(left.mu, right.mu, **close_kwargs):
            #we've reached a breaking point
            if doneLeft:
                #can't find a lower bound to efac, just return where we're at
                r = optimize.zeros.RootResults(
                    root=left.mu, iterations=i + 1, function_calls=i, flag=0)
                setattr(r, 'left', left)
                setattr(r, 'right', right)
                setattr(r, 'info', 'all close and doneLeft')
                return left, right, r

            #elif not doneLeft and not doneRight:
            else:
                #all close, and no good on either end -> no spinodal
                r = optimize.zeros.RootResults(
                    root=None, iterations=i + 1, function_calls=i, flag=1)
                setattr(r, 'left', left)
                setattr(r, 'right', right)
                setattr(r, 'doneLeft', doneLeft)
                setattr(r, 'doneRight', doneRight)
                setattr(r, 'info', 'all close and not done Left')
                return None, None, r

        mu_mid = 0.5 * (left.mu + right.mu)
        #print(mu_mid)
        mid = ref.reweight(mu_mid, **reweight_kwargs)

        v = mid.DeltabetaE_phaseIDs(**DeltabetaE_kwargs)[ID]

        if v >= efac and np.isfinite(mid.Omega_phase().sel(phase=ID).values):
            left = mid
        else:
            right = mid

    print('i', i)
    print(ID)
    print('left mu', left.mu)
    print('right mu', right.mu)
    print('dmu', left.mu - right.mu)

    print('left DE', left.DeltabetaE_phaseIDs())
    print('right DE', right.DeltabetaE_phaseIDs())

    print('doneleft', doneLeft)
    print('doneright', doneRight)
    print('close', np.allclose(left.mu, right.mu, **close_kwargs))

    raise RuntimeError('did not finish')


def _solve_spinodal(ref,
                    ID,
                    mu_in,
                    a,
                    b,
                    efac=1.0,
                    reweight_kwargs={},
                    argmax_kwargs={},
                    phases_kwargs={},
                    ftag_phases=None,
                    DeltabetaE_kwargs={},
                    **kwargs):

    idx = mu_in.index(None)
    reweight_kwargs = dict(dict(ZeroMax=True), **reweight_kwargs)

    def f(x):
        mu = mu_in[:]
        mu[idx] = x
        c = ref.reweight(mu, **reweight_kwargs)

        f.lnpi = c

        return c.DeltabetaE_phaseIDs(**DeltabetaE_kwargs)[ID] - efac

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)
    mu = f.lnpi.mu

    return mu, r, f.lnpi


def _get_step(C, ID, **kwargs):
    """
    find step value on 

    if DeltabetaE[-1,ID] - DeltabetaE[0,ID]<0 -> step=+1 (go right)
    
    else step = -1
    """

    delta = C[-1].DeltabetaE_phaseIDs(**kwargs)[ID] - \
            C[0].DeltabetaE_phaseIDs(**kwargs)[ID]

    if delta == 0:
        raise ValueError('could not determine step, delta==0')

    elif delta < 0.0:
        #go right
        return +1

    else:
        #go left
        return -1


def get_spinodal(C,
                 ID,
                 efac=1.0,
                 dmu=0.5,
                 vmin=0.0,
                 vmax=1e20,
                 ntry=20,
                 step=None,
                 nmax=20,
                 reweight_kwargs={},
                 DeltabetaE_kwargs={},
                 close_kwargs={},
                 solve_kwargs={},
                 full_output=False):
    """
    locate spinodal point for a given phaseID


    Parameters
    ----------
    C : lnPi_collection
        initial estimates to work from.  Function assumes C is in mu sorted order

    ID : int
        phaseID to work with

    mu_in : list
        list with value of static chem pot, and None for variable. e.g.,
        mu_in=[None,0.0] implies mu[0] is variable, and mu[1]=0.0

    efac : float (Default 1.0)
        cutoff value for spinodal

    dmu : float (Default 0.5)
        factor to kick back if C doesn't already have left and right bounds
    
    vmin : float (default 0.0)
        value denoting vmin, i.e., value of DeltabetaE if phaseID does not exist

    vmax : float (default 1e20)
        value indicating no transition, but phaseID==ID present

    ntry : int (Default 20)
        number of times to try kicking forward/backwards to find bracket

    step : int or None (Default None)
        if +1, step forward
        if -1, step backward 
        if None, try to determine step

    nmax : int (Default 20)
        max number of steps to refine bracket
    
    reweight_kwargs : dict
        extra arguments to reweight

    DeltabetaE_kwargs : dict
        extra arguemtns to lnPi.DeltabetaE_phaseIDs

    close_kwargs : dict
        arguments to np.allclose


    solve_kwargs : dict
        extra arguments to scipy.optimize.brentq

    full_output : bool (Default False)
        if true, return output info object


    Returns
    -------
    out : lnPi_phases object at spinodal point

    r : output info object (optional, returned if full_output is True)

    """
    assert (len(C) > 1)

    if step is None:
        step = _get_step(C, ID, **DeltabetaE_kwargs)

    if step == +1:
        CC = C
    elif step == -1:
        CC = C[-1::-1]
    else:
        raise ValueError('bad step')

    msk = C[0].mu != C[1].mu
    assert msk.sum() == 1

    mu_idx = np.where(msk)[0][0]
    mu_in = list(C[0].mu[:])
    mu_in[mu_idx] = None

    #get initial bracket
    L, R = _initial_bracket_spinodal_right(
        CC,
        ID,
        mu_in,
        efac=efac,
        dmu=dmu,
        vmax=vmax,
        ntry=ntry,
        step=step,
        reweight_kwargs=reweight_kwargs,
        DeltabetaE_kwargs=DeltabetaE_kwargs)

    left, right, rr = _refine_bracket_spinodal_right(
        L,
        R,
        ID,
        efac=efac,
        nmax=nmax,
        vmin=vmin,
        vmax=vmax,
        reweight_kwargs=reweight_kwargs,
        DeltabetaE_kwargs=DeltabetaE_kwargs)

    if left is None and right is None:
        #no spinodal found and left and right are close
        spin = None
        r = rr

    elif rr.converged:
        #converged to a solution

        spin = left

        r = rr
        setattr(r, 'bracket_iteration', rr.iterations)
        setattr(r, 'from_solve', False)

    else:
        #solve
        if step == -1:
            left, right = right, left

        a, b = left.mu[mu_idx], right.mu[mu_idx]

        mu, r, spin = _solve_spinodal(
            C[0],
            ID,
            mu_in,
            a,
            b,
            efac=efac,
            reweight_kwargs=reweight_kwargs,
            argmax_kwargs=C[0]._argmax_kwargs,
            phases_kwargs=C[0]._phases_kwargs,
            ftag_phases=C[0]._ftag_phases,
            DeltabetaE_kwargs=DeltabetaE_kwargs,
            **solve_kwargs)

        setattr(r, 'bracket_iterations', rr.iterations)
        setattr(r, 'from_solve', True)

    if full_output:
        return spin, r
    else:
        return spin
