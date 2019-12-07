"""
Set of helper utilities to work with single component system
"""

import numpy as np
from scipy.optimize import brentq


def get_lnz_min(target,
               lnz,
               C,
               build_phases,
               phase_id=0,
               component=0,
               dlnz=0.5,
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
    lnz : like with one element as None
        lnz array.  All values except index with value `None` are fixed
    C : CollectionPhases
        initial guess to work from.  This is assumed to be sorted in accending lnz order
    build_phases : callable
        funciton to build new phases objects
    phase_id : int, default=0
        phase id tag to consider
    component : int, default=0
        component to consider
    dlnz : float, default=0.5
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
    phases : Phases object as solution lnz
    solution_parameters : optional
    """

    has_idx = np.array([phase_id in x.index for x in C])
    if has_idx.sum() == 0:
        raise ValueError('no phase {}'.format(phase_id))

    # where have phase_id
    w = np.where(has_idx)[0]
    # which lnz varies
    lnz_idx = lnz.index(None)

    # input rho
    selector = dict(phase=phase_id, component=component)
    rho = C.xgce.dens.sel(**selector).values

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
        new_lnz = lnz.copy()
        new_lnz[lnz_idx] = C[w[0]].lnz[lnz_idx]
        new_lnz = np.asarray(new_lnz)

        dlnz_left = dlnz

        for i in range(ntry):
            new_lnz[lnz_idx] -= dlnz_left

            p = build_phases(ref=ref, lnz=new_lnz, **build_kws)
            if phase_id in p.index:
                if p.xgce.dens.sel(**selector).values < target:
                    left = p
                    break
            # grow dlnz
            dlnz_left += dlnz

    if left is None:
        raise RuntimeError('could not find left bounds')

    # right bracket
    right = None
    for i in w:
        if rho[i] > target:
            right = C[i]
            break

    if right is None:
        new_lnz = lnz.copy()
        new_lnz[lnz_idx] = C[w[-1]].lnz[lnz_idx]
        new_lnz = np.asarray(new_lnz)

        dlnz_right = dlnz

        for i in range(ntry):
            new_lnz[lnz_idx] += dlnz_right

            p = build_phases(ref=ref, lnz=lnz_new, **build_kws)
            if phase_id not in p.index:
                # went to far
                new_lnz[lnz_idx] -= dlnz_right
                # reset to half dlnz
                dlnz_right = dlnz * 0.5
            else:
                if p.xgce.dens.sel(**selector).values > target:
                    right = p
                    break
            dlnz_right += dlnz

    if right is None:
        raise RuntimeError('could not find right bounds')

    def f(x):
        lnz_new = lnz.copy()
        lnz_new[lnz_idx] = x
        p = build_phases(ref=ref, lnz=lnz_new, **build_kws)
        f.lnpi = p
        return p.xgce.dens.sel(**selector).values - target

    a, b = sorted([x.lnz[lnz_idx] for x in [left, right]])

    if solve_kws is None:
        solve_kws = {}

    xx, r = brentq(f, a, b, full_output=True, **solve_kws)
    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi




def _get_lnz_max(edge_distance_min,
               lnz,
               C,
               build_phases,
               ref,
               dlnz=0.5,
               dlnz_min=1e-5,
               ntry=50,
               build_kws=None,
               full_output=True):
    """
    find the lnz such that edge distance < edge_distance_min
    old version.  Use bisection below
    """

    if build_kws is None:
        build_kws = {}

    # minilnzm edge distance
    edge_distance = C.xgce.edge_distance(ref).min('phase').values
    lnz_idx = lnz.index(None)

    left = None
    w = np.where(edge_distance > edge_distance_min)[0]
    if len(w) != 0:
        # have a left bound
        left = C[w[-1]]
    else:
        new_lnz = lnz.copy()
        new_lnz[lnz_idx] = C[w[0]].lnz[lnz_idx]
        new_lnz = np.asarray(new_lnz)
        dlnz_left = dlnz

        for i in range(ntry):
            new_lnz[lnz_idx] -= dlnz_left
            p = build_phases(ref=ref, lnz=new_lnz, **build_kws)

            if p.xgce.edge_distance(ref).min(
                    'phase').values > edge_distance_min:
                left = p
                break
            dlnz_left += dlnz

    if left is None:
        raise RuntimeError('could not find left bound')


    # iteratively progress
    new_lnz = left.lnz.copy()
    dlnz_right = dlnz

    # set initial right
    right = left

    for i in range(ntry):
        new_lnz[lnz_idx] += dlnz_right
        p = build_phases(ref=ref, lnz=new_lnz, **build_kws)
        dist = p.xgce.edge_distance(ref).min('phase').values

        if dist < edge_distance_min:
            # step back
            new_lnz[lnz_idx] -= dlnz_right
            # shrink delta_lnz
            dlnz_right *= 0.5
        else:
            right = p
            if dlnz_right < dlnz_min:
                break

    if full_output:
        info = dict(niter=i, precision=dlnz_right)
        return right, info
    else:
        return right


def get_lnz_max(edge_distance_min,
               ref, build_phases,
               lnz_idx=0,
               lnz_start=None,
               C=None,
               dlnz=0.5,
               threshold_abs=1e-4,
               niter=50,
               build_kws=None,
               full_output=True):
    """
    find max lnz by bisection
    """

    if build_kws is None:
        build_kws = {}

    if lnz_start is None:
        lnz_start = ref.lnz.copy()


    # need left/right bounds
    # left is greatest lnz point with edge_distance > edge_distance_min
    # right is least lnz point with edge_distance < edge_distance_min
    left = right = None
    lnz_left = lnz_right = None

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
            lnz_left = C[0].lnz.copy()

        # right
        w = np.where(edge_distance < edge_distance_min)[0]
        if len(w) != 0:
            right = C[w[0]]
        else:
            lnz_right = C[-1].lnz.copy()


    # if left not set, try to find it
    if left is None:
        if lnz_left is None:
            lnz_left = lnz_start.copy()

        dlnz_loc = dlnz
        for i in range(niter):
            lnz_left[lnz_idx] -= dlnz_loc
            p = build_phases(ref=ref, lnz=lnz_left, **build_kws)

            if p.xgce.edge_distance(ref).min('phase').values >= edge_distance_min:
                left = p
                n_left = i
                break
            # grow dlnz
            dlnz_loc += dlnz

    if left is None:
        raise RuntimeError('could not find left bound')


    # if right not set, try to find it
    if right is None:
        if lnz_right is None:
            lnz_right = lnz_start.copy()

        dlnz_loc = dlnz
        for i in range(niter):
            lnz_right[lnz_idx] += dlnz_loc
            p = build_phases(ref=ref, lnz=lnz_right, **build_kws)

            if p.xgce.edge_distance(ref).min('phase').values < edge_distance_min:
                right = p
                n_right = i
                break
            # grow dlnz
            dlnz_loc += dlnz

    if right is None:
        raise RuntimeError('could not find right bound')



    # not do bisection
    P = [left, right]
    Y = [x.xgce.edge_distance(ref).min('phase').values for x in P]


    for i in range(niter):
        delta = np.abs(P[1].lnz[lnz_idx] - P[0].lnz[lnz_idx])
        if delta < threshold_abs:
            break
        lnz_mid = 0.5 * (P[0].lnz + P[1].lnz)
        mid = build_phases(ref=ref, lnz=lnz_mid, **build_kws)
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




