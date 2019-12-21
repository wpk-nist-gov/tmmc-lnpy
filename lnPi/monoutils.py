"""
Set of helper utilities to work with single component system
"""

import numpy as np
from scipy.optimize import brentq


def get_lnz_min(target,
                C,
                build_phases,
                phase_id=0,
                component=None,
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
        target density of specified component
    C : CollectionPhases
        initial guess to work from.  This is assumed to be sorted in accending lnz order
    build_phases : callable
        funciton to build new phases objects
    phase_id : int, default=0
        phase id tag to consider
    component : int, optional
        component to consider. If not specified, use build_phases.index
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
    lnz_idx = build_phases.index

    if component is None:
        component = lnz_idx

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
        new_lnz = C[w[0]].lnz[lnz_idx]
        dlnz_left = dlnz

        for i in range(ntry):
            new_lnz -= dlnz_left
            p = build_phases(new_lnz, ref=ref, **build_kws)
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
        new_lnz = C[w[-1]].lnz[lnz_idx]
        dlnz_right = dlnz

        for i in range(ntry):
            new_lnz += dlnz_right

            p = build_phases(new_lnz, ref=ref, **build_kws)
            if phase_id not in p.index:
                # went to far
                new_lnz -= dlnz_right
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
        lnz_new = x
        p = build_phases(lnz_new, ref=ref, **build_kws)
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
                 C,
                 build_phases,
                 ref=None,
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

    if ref is None:
        ref = build_phases.phase_creator.ref

    # minilnzm edge distance
    edge_distance = C.xgce.edge_distance(ref).min('phase').values
    lnz_idx = build_phases.index

    left = None
    w = np.where(edge_distance > edge_distance_min)[0]
    if len(w) != 0:
        # have a left bound
        left = C[w[-1]]
    else:
        new_lnz = C[w[0]].lnz[lnz_idx]
        dlnz_left = dlnz

        for i in range(ntry):
            new_lnz -= dlnz_left
            p = build_phases(new_lnz, ref=ref, **build_kws)

            if p.xgce.edge_distance(ref).min(
                    'phase').values > edge_distance_min:
                left = p
                break
            dlnz_left += dlnz

    if left is None:
        raise RuntimeError('could not find left bound')

    # iteratively progress
    new_lnz = left.lnz[lnz_idx]
    dlnz_right = dlnz

    # set initial right
    right = left

    for i in range(ntry):
        new_lnz += dlnz_right
        p = build_phases(new_lnz, ref=ref, **build_kws)
        dist = p.xgce.edge_distance(ref).min('phase').values

        if dist < edge_distance_min:
            # step back
            new_lnz -= dlnz_right
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
                build_phases,
                ref=None,
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
    if ref is None:
        ref = build_phases.phase_creator.ref

    lnz_idx = build_phases.index
    if lnz_start is None:
        lnz_start = ref.lnz[lnz_idx]

    # need left/right bounds
    # left is greatest lnz point with edge_distance > edge_distance_min
    # right is least lnz point with edge_distance < edge_distance_min
    left = right = None
    lnz_left = lnz_right = None
    n_left = n_right = None

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
            lnz_left = C[0].lnz[lnz_idx]

        # right
        w = np.where(edge_distance < edge_distance_min)[0]
        if len(w) != 0:
            right = C[w[0]]
        else:
            lnz_right = C[-1].lnz[lnz_idx]

    # if left not set, try to find it
    if left is None:
        if lnz_left is None:
            lnz_left = lnz_start

        dlnz_loc = dlnz
        for i in range(niter):
            lnz_left -= dlnz_loc
            p = build_phases(lnz_left, ref=ref, **build_kws)
            if p.xgce.edge_distance(ref).min(
                    'phase').values >= edge_distance_min:
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
            lnz_right = lnz_start

        dlnz_loc = dlnz
        for i in range(niter):
            lnz_right += dlnz_loc
            p = build_phases(lnz_right, ref=ref, **build_kws)

            if p.xgce.edge_distance(ref).min(
                    'phase').values < edge_distance_min:
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
        lnz_mid = 0.5 * (P[0].lnz[lnz_idx] + P[1].lnz[lnz_idx])
        mid = build_phases(lnz_mid, ref=ref, **build_kws)
        y_mid = mid.xgce.edge_distance(ref).min('phase').values

        if y_mid >= edge_distance_min:
            index = 0
        else:
            index = 1

        P[index] = mid
        Y[index] = y_mid

    left, right = P
    if full_output:
        info = dict(niter=i,
                    niter_left=n_left,
                    niter_right=n_right,
                    precision=delta)
        return left, info
    else:
        return left


from .core import CollectionPhases
from functools import partial

def limited_collection(build_phases,
                       lnz_range, dlnz,
                       course_step=None,
                       edge_distance_min=None, rho_min=None,
                       lnz_min_kws=None, lnz_max_kws=None,
                       ref=None, build_phases_kws=None, build_stability_kws=None, nmax=None,
                       xarray_output=True, collection_kws=None):

    lnzs = np.arange(lnz_range[0], lnz_range[1] + dlnz * 0.5, dlnz)


    if course_step is None:
        course_step = max(len(lnzs) // 40, 1)

    if collection_kws is None:
        collection_kws = {}


    get_collection = partial(
        CollectionPhases.from_builder,
        build_phases=build_phases,
        build_phases_kws=build_phases_kws, nmax=nmax, xarray_output=xarray_output,
        **collection_kws
    )

    # limit lnz
    c_course = None
    lnz_min, lnz_max = lnzs[0], lnzs[-1]
    if edge_distance_min is not None or rho_min is not None:
        c_course = get_collection(lnzs[::course_step])

        if rho_min is not None:
            try:
                if lnz_min_kws is None:
                    lnz_min_kws = {}
                p_min, o = get_lnz_min(rho_min, c_course, build_phases,
                                       build_kws=build_phases_kws, **lnz_min_kws)
                if o.converged:
                    lnz_min = p_min.lnz[build_phases.index] - dlnz
            except:
                pass

        if edge_distance_min is not None:
            try:
                if lnz_max_kws is None:
                    lnz_max_kws = {}
                p_max, o = get_lnz_max(edge_distance_min, build_phases,
                                       build_kws=build_phases_kws, C=c_course, **lnz_max_kws)
                lnz_max = p_max.lnz[build_phases.index]
            except:
                pass
    lnzs = lnzs[(lnzs >= lnz_min) & (lnzs <= lnz_max)]


    c = get_collection(lnzs)
    if c_course is None:
        c_course = c
    return c_course, c






