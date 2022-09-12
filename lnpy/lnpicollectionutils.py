"""
Set of helper utilities to work with single component system
"""

# from functools import partial

import numpy as np
from scipy.optimize import brentq

from .lnpiseries import lnPiCollection


def get_lnz_min(
    target,
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
    full_output=True,
):
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
        phase id tag to consider.  If None, consider min over 'phases'
    component : int, optional
        component to consider.
        If component is None, use build_phases.index.
        If component=='None', then consider total density
    dlnz : float, default=0.5
        how to change the chemical potential if `C` doesn't already
        bracket solution
    ref : lnPiMasked, optional
        optional lnPiMasked object to pass to build_phases
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

    if phase_id is not None and phase_id not in C.index.get_level_values("phase"):
        raise ValueError("no phase {}".format(phase_id))

    lnz_idx = build_phases.index
    if isinstance(component, str) and component.lower() == "none":

        def getter_comp(x):
            return x.xge.dens_tot

    else:
        if component is None:
            component = lnz_idx

        def getter_comp(x):
            return x.xge.dens.sel(component=component)

    if phase_id is None:

        def getter(x):
            return getter_comp(x).min("phase")

    else:

        def getter(x):
            return getter_comp(x).sel(phase=phase_id)

    s = getter(C).to_series().dropna()
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
            if phase_id in p._get_level("phase") and getter(p).values < target:
                left = p
                break
            dlnz_left *= dfac

    if left is None:
        raise RuntimeError("could not find left bounds")

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

            if phase_id not in p._get_level("phase"):
                # went to far
                new_lnz -= dlnz_right
                # reset to half dlnz
                dlnz_right = dlnz_right * 0.5
            elif getter(p).values > target:
                right = p
                break
            else:
                dlnz_right *= dfac

    if right is None:
        raise RuntimeError("could not find right bounds")

    def f(x):
        lnz_new = x
        p = build_phases(lnz_new, ref=ref, **build_kws)
        f.lnpi = p
        return getter(p).values - target

    a, b = sorted([x._get_lnz(lnz_idx) for x in [left, right]])
    if solve_kws is None:
        solve_kws = {}
    xx, r = brentq(f, a, b, full_output=True, **solve_kws)
    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi


def get_lnz_max(
    edge_distance_min,
    build_phases,
    ref=None,
    lnz_start=None,
    C=None,
    dlnz=0.5,
    dfac=1.0,
    threshold_abs=1e-4,
    ntry=50,
    build_kws=None,
    full_output=True,
):
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

    def getter(p):
        v = p.xge.edge_distance(ref)
        if not p._xarray_unstack:
            v = v.unstack(p._concat_dim)
        return v.min("phase")

    # if have C, try to use it
    if C is not None:
        # see if bound contained in C

        s = getter(C).to_series()

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
        for i in range(ntry):
            lnz_left -= dlnz_loc
            p = build_phases(lnz_left, ref=ref, **build_kws)
            if getter(p).values >= edge_distance_min:
                left = p
                n_left = i
                break
            # grow dlnz
            dlnz_loc *= dfac

    if left is None:
        raise RuntimeError("could not find left bound")

    # if right not set, try to find it
    if right is None:
        if lnz_right is None:
            lnz_right = lnz_start

        dlnz_loc = dlnz
        for i in range(ntry):
            lnz_right += dlnz_loc
            p = build_phases(lnz_right, ref=ref, **build_kws)

            if getter(p).values < edge_distance_min:
                right = p
                n_right = i
                break
            # grow dlnz
            dlnz_loc *= dfac

    if right is None:
        raise RuntimeError("could not find right bound")

    # not do bisection
    P = [left, right]
    Y = [getter(x).values for x in P]

    for i in range(ntry):
        lnz = [x._get_lnz(lnz_idx) for x in P]
        delta = np.abs(lnz[1] - lnz[0])
        if delta < threshold_abs:
            break
        lnz_mid = 0.5 * (lnz[0] + lnz[1])
        mid = build_phases(lnz_mid, ref=ref, **build_kws)
        y_mid = getter(mid).values

        if y_mid >= edge_distance_min:
            index = 0
        else:
            index = 1

        P[index] = mid
        Y[index] = y_mid

    left, right = P
    if full_output:
        info = dict(ntry=i, ntry_left=n_left, ntry_right=n_right, precision=delta)
        return left, info
    else:
        return left


class _BaseLimit(object):
    _bound_side = None  # -1 for lower bound, +1 for upper bound

    def __init__(self, parent):
        self._parent = parent

    @property
    def parent(self):
        return self._parent

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("to be implemented in subclass")

    @property
    def lnz_idx(self):
        return self._lnz_idx

    @lnz_idx.setter
    def lnz_idx(self, i):
        self._lnz_idx = i

    @property
    def access(self):
        return self._access

    @access.setter
    def access(self, val):
        if not isinstance(val, lnPiCollection):
            raise ValueError("access should be lnPiCollection")
        self._access = val

    def _get_bound(self, sign, access=None, lnz_idx=None):
        if access is None:
            access = self.access
        if lnz_idx is None:
            lnz_idx = self.lnz_idx
        if sign == self._bound_side:
            return access._get_lnz(lnz_idx)
        else:
            return sign * np.inf

    def get_bounds(self, access=None, lnz_idx=None):
        return np.array(
            [
                self._get_bound(sign=sign, access=access, lnz_idx=lnz_idx)
                for sign in [-1, +1]
            ]
        )

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        assert len(bounds) == 2
        self._bounds = np.array(bounds)

    def set_bounds(self, access=None, lnz_idx=None):
        self.bounds = self.get_bounds(access=access, lnz_idx=lnz_idx)

    @property
    def lb(self):
        return self.bounds[0]

    @property
    def ub(self):
        return self.bounds[1]

    @property
    def lnz_list(self):
        return self._lnz_list

    def get_lnz_list(self, lnz=None, lnz_idx=None):
        if lnz is None:
            lnz = self.access.iloc[0].lnz
        if lnz_idx is None:
            lnz_idx = self.lnz_idx
        lnz = list(lnz)
        lnz[lnz_idx] = None
        return lnz

    def set_lnz_list(self, lnz=None, lnz_idx=None):
        self._lnz_list = self.get_lnz_list(lnz, lnz_idx)


@lnPiCollection.decorate_accessor("bounds_lower", single_create=True)
class LowerBounds(_BaseLimit):

    _bound_side = -1

    def __call__(
        self,
        dens_min,
        build_phases,
        phase_id=0,
        component=None,
        dlnz=0.5,
        dfac=1.0,
        ref=None,
        build_kws=None,
        ntry=20,
        solve_kws=None,
        inplace=True,
        force=False,
    ):

        if hasattr(self, "_access") and not force:
            p = self.access
            info = self._info
            lnz_idx = self.lnz_idx
            bounds = self.bounds

        else:
            lnz_idx = build_phases.index
            p, info = get_lnz_min(
                target=dens_min,
                C=self._parent,
                build_phases=build_phases,
                phase_id=phase_id,
                component=component,
                dlnz=dlnz,
                dfac=1.0,
                ref=ref,
                build_kws=build_kws,
                ntry=ntry,
                solve_kws=solve_kws,
            )

            bounds = self.get_bounds(p, lnz_idx)

        if inplace:
            self.access = p
            self._info = info
            self.lnz_idx = lnz_idx
            self.bounds = bounds
            self.set_lnz_list()
        else:
            return bounds, p, lnz_idx, info


@lnPiCollection.decorate_accessor("bounds_upper", single_create=True)
class UpperBounds(_BaseLimit):
    _bound_side = +1

    def __call__(
        self,
        edge_distance_min,
        build_phases,
        ref=None,
        lnz_start=None,
        dlnz=0.5,
        dfac=1.0,
        threshold_abs=1e-4,
        ntry=20,
        build_kws=None,
        inplace=True,
        force=False,
    ):

        if hasattr(self, "_access") and not force:
            p = self.access
            info = self._info
            lnz_idx = self.lnz_idx
            bounds = self.bounds

        else:
            lnz_idx = build_phases.index
            p, info = get_lnz_max(
                edge_distance_min=edge_distance_min,
                C=self._parent,
                build_phases=build_phases,
                lnz_start=None,
                dlnz=dlnz,
                dfac=1.0,
                threshold_abs=threshold_abs,
                ntry=ntry,
                ref=ref,
                build_kws=build_kws,
                full_output=True,
            )
            bounds = self.get_bounds(p, lnz_idx)

        if inplace:
            self.access = p
            self._info = info
            self.lnz_idx = lnz_idx
            self.bounds = bounds
            self.set_lnz_list()
        else:
            return bounds, p, lnz_idx, info


def build_grid(
    x=None,
    dx=None,
    x_range=None,
    x0=None,
    offsets=None,
    even_grid=False,
    digits=None,
    unique=True,
    outlier=False,
):
    """
    build a grid of values


    Parameters
    ----------
    x : array, optional
        if passes, use this as the base array
    dx : float, optional
        delta x. See below for uses
    x_range : array of length 2, optional
        if x not specified, make an array with
        `x=np.arange(x_range[0], x_range[1] + dx/2, dx)`
    x0 : float, optional
        see offsets
    offsets : array of length 2, optional
        if x or x_range not specified, set x_range = x0 + offsets
    even_grid : bool, default=True
        if True, make sure all values are multiples of dx
    digits : int, optional
        if passes, round output to this many digits
    outlier : bool, default=False
        if outlier is True, allow values outsize x

    Returns
    -------
    out : array of values
    """

    if x is None:
        assert dx is not None

        if x_range is None:
            x_range = x0 + np.array(offsets)
        assert len(x_range) == 2
        x = np.arange(x_range[0], x_range[1] + dx * 0.5, dx)

    if not outlier:
        lb, ub = x.min(), x.max()

    if even_grid:
        x = np.round(x / dx) * dx

    if digits is not None:
        x = np.round(x, digits)

    if unique:
        x = np.unique(x)

    if not outlier:
        x = x[(lb <= x) & (x <= ub)]

    return x


def limited_collection(
    build_phases,
    dlnz,
    lnz_range=None,
    offsets=None,
    digits=None,
    even_grid=True,
    course_step=None,
    edge_distance_min=None,
    dens_min=None,
    lnz_min_kws=None,
    lnz_max_kws=None,
    ref=None,
    build_kws=None,
    build_stability_kws=None,
    nmax=None,
    xarray_output=True,
    collection_kws=None,
    limit_course=False,
):
    """
    build a lnPiCollection over a range of lnz values

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

    # get_collection = partial(
    #     lnPiCollection.from_builder,
    #     build_phases=build_phases,
    #     build_kws=build_kws, nmax=nmax, xarray_output=xarray_output,
    #     **collection_kws
    # )

    # limit lnz
    c_course = None
    lnz_min, lnz_max = lnzs[0], lnzs[-1]
    if edge_distance_min is not None or dens_min is not None:
        # c_course = get_collection(lnzs[::course_step])
        c_course = lnPiCollection.from_builder(
            lnzs[::course_step],
            build_phases=build_phases,
            build_kws=build_kws,
            nmax=nmax,
            xarray_output=xarray_output,
            **collection_kws,
        )

        if dens_min is not None:
            try:
                if lnz_min_kws is None:
                    lnz_min_kws = {}
                p_min, o = get_lnz_min(
                    dens_min, c_course, build_phases, build_kws=build_kws, **lnz_min_kws
                )
                if o.converged:
                    lnz_min = p_min.iloc[0].lnz[build_phases.index] - dlnz
            except Exception:
                pass

        if edge_distance_min is not None:
            try:
                if lnz_max_kws is None:
                    lnz_max_kws = {}
                p_max, o = get_lnz_max(
                    edge_distance_min,
                    build_phases,
                    build_kws=build_kws,
                    C=c_course,
                    **lnz_max_kws,
                )
                lnz_max = p_max.iloc[0].lnz[build_phases.index]
            except Exception:
                pass

    lnzs = lnzs[(lnzs >= lnz_min) & (lnzs <= lnz_max)]
    c = lnPiCollection.from_builder(
        lnzs,
        build_phases=build_phases,
        build_kws=build_kws,
        nmax=nmax,
        xarray_output=xarray_output,
        **collection_kws,
    )

    if c_course is None:
        c_course = c
    elif limit_course:
        slnz = "lnz_{}".format(build_phases.index)
        q = f"{slnz} >= {lnz_min} and {slnz} <= {lnz_max}"
        c_course = c_course.query(q)

    return c_course, c
