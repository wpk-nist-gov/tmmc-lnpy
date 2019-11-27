"""
Set of helper utilities to work with single component system
"""

import numpy as np
from scipy.optimize import brentq


def get_mu_min(target,
               mu,
               C,
               build_phases,
               phase_id=0,
               component=0,
               dmu=0.5,
               ref=None,
               build_kws=None,
               ntry=20,
               solve_kws=None,
               full_output=True):
    """
    solve for target density

    Parameters
    ----------
    target : float
        target density
    mu : like with one element as None
        mu array.  All values except index with value `None` are fixed
    C : CollectionPhases
        initial guess to work from.  This is assumed to be sorted in accending mu order
    build_phases : callable
        funciton to build new phases objects
    phase_id : int, default=0
        phase id tag to consider
    component : int, default=0
        component to consider
    dmu : float, default=0.5
        how to change the chemical potential if `C` doesn't already
        bracket solution
    ref : MaskedlnPi, optional
        optional MaskedlnPi object to pass to build_phases
    build_kws : dictionary, optional
        optional arguments to `build_phases`
    ntry : int, default=20
        number of times to attempted finding left/right bracketing objects
    solve_kws : dictionary, optional
        optional arguments to `scipy.optimize.brentq`
    full_output : bool, default=True
        if True, return solve parameters

    Returns
    -------
    phases : Phases object as solution mu
    solution_parameters : optional
    """

    has_idx = np.array([phase_id in x.index for x in C])
    if has_idx.sum() == 0:
        raise ValueError('no phase {}'.format(phase_id))

    # where have phase_id
    w = np.where(has_idx)[0]
    # which mu varies
    mu_idx = mu.index(None)

    # input rho
    selector = dict(phase=phase_id, component=component)
    rho = C.xgce.density.sel(**selector).values

    # builder
    if build_kws is None:
        build_kws = {}

    # left bracket
    left = None
    for i in w[-1::-1]:
        if rho[i] < target:
            left = C[i]
            break

    if left is None:
        new_mu = mu.copy()
        new_mu[mu_idx] = C[w[0]].mu[mu_idx]
        new_mu = np.asarray(new_mu)

        dmu_left = dmu

        for i in range(ntry):
            new_mu[mu_idx] -= dmu_left

            p = build_phases(ref=ref, mu=new_mu, **build_kws)
            if phase_id in p.index:
                if p.xgce.density.sel(**selector).values < target:
                    left = p
                    break
            # grow dmu
            dmu_left += dmu

    if left is None:
        raise RuntimeError('could not find left bounds')

    # right bracket
    right = None
    for i in w:
        if rho[i] > target:
            right = C[i]
            break

    if right is None:
        new_mu = mu.copy()
        new_mu[mu_idx] = C[w[-1]].mu[mu_idx]
        new_mu = np.asarray(new_mu)

        dmu_right = dmu

        for i in range(ntry):
            new_mu[mu_idx] += dmu_right

            p = build_phases(ref=ref, mu=mu_new, **build_kws)
            if phase_id not in p.index:
                # went to far
                new_mu[mu_idx] -= dmu_right
                # reset to half dmu
                dmu_right = dmu * 0.5
            else:
                if p.xgce.density.sel(**selector).values > target:
                    right = p
                    break
            dmu_right += dmu

    if right is None:
        raise RuntimeError('could not find right bounds')

    def f(x):
        mu_new = mu.copy()
        mu_new[mu_idx] = x
        p = build_phases(ref=ref, mu=mu_new, **build_kws)
        f.lnpi = p
        return p.xgce.density.sel(**selector).values - target

    a, b = sorted([x.mu[mu_idx] for x in [left, right]])

    if solve_kws is None:
        solve_kws = {}

    xx, r = brentq(f, a, b, full_output=True, **solve_kws)
    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi




def _get_mu_max(edge_distance_min,
               mu,
               C,
               build_phases,
               ref,
               dmu=0.5,
               dmu_min=1e-5,
               ntry=50,
               build_kws=None,
               full_output=True):
    """
    find the mu such that edge distance < edge_distance_min
    old version.  Use bisection below
    """

    if build_kws is None:
        build_kws = {}

    # minimum edge distance
    edge_distance = C.xgce.edge_distance(ref).min('phase').values
    mu_idx = mu.index(None)

    left = None
    w = np.where(edge_distance > edge_distance_min)[0]
    if len(w) != 0:
        # have a left bound
        left = C[w[-1]]
    else:
        new_mu = mu.copy()
        new_mu[mu_idx] = C[w[0]].mu[mu_idx]
        new_mu = np.asarray(new_mu)
        dmu_left = dmu

        for i in range(ntry):
            new_mu[mu_idx] -= dmu_left
            p = build_phases(ref=ref, mu=new_mu, **build_kws)

            if p.xgce.edge_distance(ref).min(
                    'phase').values > edge_distance_min:
                left = p
                break
            dmu_left += dmu

    if left is None:
        raise RuntimeError('could not find left bound')


    # iteratively progress
    new_mu = left.mu.copy()
    dmu_right = dmu

    # set initial right
    right = left

    for i in range(ntry):
        new_mu[mu_idx] += dmu_right
        p = build_phases(ref=ref, mu=new_mu, **build_kws)
        dist = p.xgce.edge_distance(ref).min('phase').values

        if dist < edge_distance_min:
            # step back
            new_mu[mu_idx] -= dmu_right
            # shrink delta_mu
            dmu_right *= 0.5
        else:
            right = p
            if dmu_right < dmu_min:
                break

    if full_output:
        info = dict(niter=i, precision=dmu_right)
        return right, info
    else:
        return right


def get_mu_max(edge_distance_min,
               ref, build_phases,
               mu_idx=0,
               mu_start=None,
               C=None,
               dmu=0.5,
               threshold_abs=1e-4,
               niter=50,
               build_kws=None,
               full_output=True):
    """
    find max mu by bisection
    """

    if build_kws is None:
        build_kws = {}

    if mu_start is None:
        mu_start = ref.mu.copy()


    # need left/right bounds
    # left is greatest mu point with edge_distance > edge_distance_min
    # right is least mu point with edge_distance < edge_distance_min
    left = right = None
    mu_left = mu_right = None

    n_left =  n_right = None

    # if have C, try to use it
    if C is not None:
        # see if bound contained in C
        edge_distance = C.xgce.edge_distance(ref).min('phase').values

        # left
        w = np.where(edge_distance > edge_distance_min)[0]
        if len(w) != 0:
            left = C[w[-1]]
        else:
            # no points with dist  > dist_min
            mu_left = C[0].mu.copy()

        # right
        w = np.where(edge_distance < edge_distance_min)[0]
        if len(w) != 0:
            right = C[w[0]]
        else:
            mu_right = C[-1].mu.copy()


    # if left not set, try to find it
    if left is None:
        if mu_left is None:
            mu_left = mu_start.copy()

        dmu_loc = dmu
        for i in range(niter):
            mu_left[mu_idx] -= dmu_loc
            p = build_phases(ref=ref, mu=mu_left, **build_kws)

            if p.xgce.edge_distance(ref).min('phase').values >= edge_distance_min:
                left = p
                n_left = i
                break
            # grow dmu
            dmu_loc += dmu

    if left is None:
        raise RuntimeError('could not find left bound')


    # if right not set, try to find it
    if right is None:
        if mu_right is None:
            mu_right = mu_start.copy()

        dmu_loc = dmu
        for i in range(niter):
            mu_right[mu_idx] += dmu_loc
            p = build_phases(ref=ref, mu=mu_right, **build_kws)

            if p.xgce.edge_distance(ref).min('phase').values < edge_distance_min:
                right = p
                n_right = i
                break
            # grow dmu
            dmu_loc += dmu

    if right is None:
        raise RuntimeError('could not find right bound')



    # not do bisection
    P = [left, right]
    Y = [x.xgce.edge_distance(ref).min('phase').values for x in P]


    for i in range(niter):
        delta = np.abs(P[1].mu[mu_idx] - P[0].mu[mu_idx])
        if delta < threshold_abs:
            break
        mu_mid = 0.5 * (P[0].mu + P[1].mu)
        mid = build_phases(ref=ref, mu=mu_mid, **build_kws)
        y_mid = mid.xgce.edge_distance(ref).min('phase').values

        if y_mid >= edge_distance_min:
            index = 0
        else:
            index = 1

        P[index] = mid
        Y[index] = y_mid

    left, right = P
    if full_output:
        info = dict(niter=i, niter_left=n_left, niter_right=n_right, precision=delta)
        return left,  info
    else:
        return left




