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
                dfac=1.0,
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

    if phase_id not in C.index.get_level_values('phase'):
        raise ValueError('no phase {}'.format(phase_id))


    lnz_idx = build_phases.index
    if component is None:
        component = lnz_idx
    # input rho
    selector = dict(phase=phase_id, component=component)

    s = C.xgce.dens.sel(**selector).to_series().dropna()
    # builder
    if build_kws is None:
        build_kws = {}

    # left bracket
    left = None
    ss = s[s < target]
    if len(ss) > 0:
        left = C.mloc[ss.index[[-1]]]
    else:
        new_lnz = C.mloc[s.index[[0]]]._get_lnz(lnz_idx)
        dlnz_left = dlnz
        for i in range(ntry):
            new_lnz -= dlnz_left
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if phase_id in p._get_level('phase') and \
               p.xgce.dens.sel(**selector).values < target:
                left = p
                break
            dlnz_left *= dfac

    if left is None:
        raise RuntimeError('could not find left bounds')

    # right bracket
    right = None
    ss = s[s > target]
    if len(ss) > 0:
        right = C.mloc[ss.index[[0]]]
    else:
        new_lnz = C.mloc[s.index[[-1]]]._get_lnz(lnz_idx)
        dlnz_right = dlnz

        for i in range(ntry):
            new_lnz += dlnz_right
            p = build_phases(new_lnz, ref=ref, **build_kws)

            if phase_id not in p._get_level('phase'):
                # went to far
                new_lnz -= dlnz_right
                # reset to half dlnz
                dlnz_right = dlnz_right * 0.5
            elif p.xgce.dens.sel(**selector).values > target:
                right = p
                break
            else:
                dlnz_right *= dfac

    if right is None:
        raise RuntimeError('could not find right bounds')

    def f(x):
        lnz_new = x
        p = build_phases(lnz_new, ref=ref, **build_kws)
        f.lnpi = p
        return p.xgce.dens.sel(**selector).values - target


    a, b = sorted([x._get_lnz(lnz_idx) for x in [left, right]])
    if solve_kws is None:
        solve_kws = {}
    xx, r = brentq(f, a, b, full_output=True, **solve_kws)
    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi


def get_lnz_max(edge_distance_min,
                build_phases,
                ref=None,
                lnz_start=None,
                C=None,
                dlnz=0.5,
                dfac=1.0,
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
        s = C.xgce.edge_distance(ref).min('phase').to_series()

        # left
        ss = s[s > edge_distance_min]
        if len(ss) > 0:
            left = C.mloc[ss.index[[-1]]]
        else:
            lnz_left = C.zloc[[0]]._get_lnz(lnz_idx)

        # right
        ss = s[s < edge_distance_min]
        if len(ss) > 0:
            right = C.mloc[ss.index[[0]]]
        else:
            lnz_right = C.zloc[[-1]]._get_lnz(lnz_idx)

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
            dlnz_loc *= dfac

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
            dlnz_loc *= dfac

    if right is None:
        raise RuntimeError('could not find right bound')

    # not do bisection
    P = [left, right]
    Y = [x.xgce.edge_distance(ref).min('phase').values for x in P]


    for i in range(niter):
        lnz = [x._get_lnz(lnz_idx) for x in P]
        delta = np.abs(lnz[1] - lnz[0])
        if delta < threshold_abs:
            break
        lnz_mid = 0.5 * (lnz[0] + lnz[1])
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


from functools import partial
from .serieswrapper import CollectionlnPi

def limited_collection(build_phases, dlnz,
                       lnz_range=None,
                       offsets=None,
                       digits=None, even_grid=True,
                       course_step=None,
                       edge_distance_min=None, rho_min=None,
                       lnz_min_kws=None, lnz_max_kws=None,
                       ref=None, build_phases_kws=None, build_stability_kws=None, nmax=None,
                       xarray_output=True, collection_kws=None, limit_course=False):
    """
    build a CollectionlnPi over a range of lnz values

    Parameters
    ----------
    build_phases : 
    """


    if lnz_range is None:
        if ref is None:
            _ref = build_phases.phase_creator.ref
        else:
            _ref = ref
        lnz_range = _ref.lnz[build_phases.index] + np.array(offsets)

    lnz_range = np.array(lnz_range)
    if even_grid:
        new_range = np.round(lnz_range / dlnz) * dlnz
        if new_range[0] < lnz_range[0]:
            new_range[0] += dlnz
        if new_range[1] > lnz_range[1]:
            new_range[1] -= dlnz
        lnz_range = new_range
    lnzs = np.arange(lnz_range[0], lnz_range[1] + dlnz * 0.5, dlnz)
    if even_grid:
        lnzs = np.round(lnzs / dlnz) * dlnz
    if digits is not None:
        lnzs = np.round(lnzs, digits)

    if course_step is None:
        course_step = max(len(lnzs) // 40, 1)
    if collection_kws is None:
        collection_kws = {}


    get_collection = partial(
        CollectionlnPi.from_builder,
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
                    lnz_min = p_min.iloc[0].lnz[build_phases.index] - dlnz
            except:
                pass

        if edge_distance_min is not None:
            try:
                if lnz_max_kws is None:
                    lnz_max_kws = {}
                p_max, o = get_lnz_max(edge_distance_min, build_phases,
                                       build_kws=build_phases_kws, C=c_course, **lnz_max_kws)
                lnz_max = p_max.iloc[0].lnz[build_phases.index]
            except:
                pass


    lnzs = lnzs[(lnzs >= lnz_min) & (lnzs <= lnz_max)]
    c = get_collection(lnzs)

    if c_course is None:
        c_course = c
    elif limit_course:
        slnz = 'lnz_{}'.format(build_phases.index)
        q = f'{slnz} >= {lnz_min} and {slnz} <= {lnz_max}'
        c_course = c_course.query(q)

    return c_course, c






