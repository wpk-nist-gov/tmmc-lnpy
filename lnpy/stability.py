"""
thermodynamic stability routines (i.e., spinodal/binodal)
"""
import itertools

import numpy as np
from scipy import optimize

from .cached_decorators import gcached
from .lnpiseries import lnPiCollection


# ###############################################################################
# Spinodal routines
def _initial_bracket_spinodal_right(
    C,
    build_phases,
    idx,
    idx_nebr=None,
    efac=1.0,
    dlnz=0.5,
    dfac=1.0,
    vmax=1e5,
    ntry=20,
    step=+1,
    ref=None,
    build_kws=None,
):
    """
    find initial bracketing lnpi_phases of phaseID==ID bracketing point where
    DeltabetaE_phaseIDS()[ID]==efac

    Parameters
    ----------
    ref : lnPiMasked, optional
    build_phases : callable
        scalar funciton to build phases
    C : lnPi_collection
        initial estimates to work from
    idx, idx_nebr : int
        id's of from/to phases.
    lnz_in : list
        list with value of static chem pot, and None for variable. e.g.,
        lnz_in=[None,0.0] implies lnz[0] is variable, and lnz[1]=0.0
    efac : float (Default 1.0)
        cutoff value for spinodal
    dlnz : float (Default 0.5)
        factor to kick back if C doesn't already have left and right bounds
    vmax : float (default 1e20)
        value indicating no transition, but phaseID==ID present
    ntry : int (Default 20)
        number of times to try kicking forward/backwards to find bracket
    step : int (Default +1)
        if +1, step forward
        if -1, step backward
    build_phases : callable, optional.
        function that returns a Phases object.
    build_kws : dict
        extra arguments to build_phases
    Returns
    -------
    left,right: lnpi_phases
        left and right bracketing lnPi_phases objects

    """

    if build_kws is None:
        build_kws = {}
    if efac <= 0:
        raise ValueError("efac must be positive")

    # Series representation of dw
    s = C.wfe.get_dw(idx, idx_nebr)
    if step < 0:
        s = s.iloc[-1::-1]

    # see if contain "left" bounds
    left = None
    s_idx = s[s > 0.0]
    if len(s_idx) == 0:
        raise ValueError("no phase {}".format(idx))

    ss = s_idx[s_idx > efac]
    if len(ss) > 0:
        # get last one
        left = C.mloc[ss.index[[-1]]]
    else:
        new_lnz = C.mloc[s_idx.index[[0]]]._get_lnz(build_phases.index)
        for i in range(ntry):
            new_lnz -= step * dlnz
            t = build_phases(new_lnz, ref=ref, **build_kws)
            if (
                idx in t._get_level("phase")
                and t.wfe_phases.get_dw(idx, idx_nebr) > efac
            ):
                left = t
                break

    if left is None:
        raise RuntimeError("could not find left")

    # right
    right = None
    ss = s[s < efac]
    if len(ss) > 0:
        right = C.mloc[ss.index[[0]]]
    else:
        new_lnz = C.mloc[s.index[[-1]]]._get_lnz(build_phases.index)
        dlnz_ = dlnz
        for i in range(ntry):
            new_lnz += step * dlnz_
            t = build_phases(new_lnz, ref=ref, **build_kws)
            if (
                idx not in t._get_level("phase")
                or t.wfe_phases.get_dw(idx, idx_nebr) < efac
            ):
                right = t
                break
            dlnz_ *= dfac

    if right is None:
        raise RuntimeError("could not find right")
    return left, right


def _refine_bracket_spinodal_right(
    left,
    right,
    build_phases,
    idx,
    idx_nebr=None,
    efac=1.0,
    nmax=30,
    vmax=1e5,
    vmin=0.0,
    ref=None,
    build_kws=None,
    close_kws=None,
):
    """
    find refined bracket with efac<DeltabetaE_left<vmax and
    vmin<DeltabetaE_right<efac

    Parameters
    ----------
    left, right : Phases objects
        left and right initial brackets
    idx, idx_nebr : int
        from/to phase id's
    efac : float (Default 1.0)
        cutoff value for spinodal
    nmax : int (Default 30)
        max number of interations
    vmin,vmax : see above
    build_phases : callable
    build_kws : dict
    close_kwargs : dict
        arguments to np.allclose
    Returns
    -------
    left,right : lnpi_phases objects
        left and right phases bracketing spinodal

    r : scipy.optimize.zeros.RootResults object
    """

    doneLeft = False
    doneRight = False
    if build_kws is None:
        build_kws = {}
    if close_kws is None:
        close_kws = {}

    for i in range(nmax):
        # if idx in left.index and idx_nebr in left.index:
        # dw = _get_dw(left, idx, idx_nebr)
        dw = left.wfe_phases.get_dw(idx, idx_nebr)
        if dw < vmax and dw > efac:
            doneLeft = True

        # dw = _get_dw(right, idx, idx_nebr)
        dw = right.wfe_phases.get_dw(idx, idx_nebr)
        if dw > vmin and dw < efac:
            doneRight = True

        #########
        # checks
        if doneLeft and doneRight:
            # find bracket
            r = optimize.RootResults(root=None, iterations=i, function_calls=i, flag=1)
            return left, right, r

        ########
        # converged?
        if np.allclose(left._get_lnz(), right._get_lnz(), **close_kws):
            # we've reached a breaking point
            if doneLeft:
                # can't find a lower bound to efac, just return where we're at
                r = optimize.RootResults(
                    root=left._get_lnz(), iterations=i + 1, function_calls=i, flag=0
                )
                for k, val in [
                    ("left", left),
                    ("right", right),
                    ("doneleft", doneLeft),
                    ("doneright", "doneRight"),
                    ("info", "all close and doneleft"),
                ]:
                    setattr(r, k, val)
                return left, right, r

            # elif not doneLeft and not doneRight:
            else:
                # all close, and no good on either end -> no spinodal
                r = optimize.RootResults(
                    root=None, iterations=i + 1, function_calls=i, flag=1
                )
                for k, val in [
                    ("left", left),
                    ("right", right),
                    ("doneleft", doneLeft),
                    ("doneright", "doneRight"),
                    ("info", "all close and doneleft"),
                ]:
                    setattr(r, k, val)
                for k, val in [
                    ("left", left),
                    ("right", right),
                    ("info", "all close and not doneleft"),
                ]:
                    setattr(r, k, val)
                return None, None, r

        # mid point phases
        lnz_mid = 0.5 * (
            left._get_lnz(build_phases.index) + right._get_lnz(build_phases.index)
        )

        mid = build_phases(lnz_mid, ref=ref, **build_kws)
        dw = mid.wfe_phases.get_dw(idx, idx_nebr)

        if (idx in mid._get_level("phase")) and (
            mid.wfe_phases.get_dw(idx, idx_nebr) >= efac
        ):
            left = mid
        else:
            right = mid

    raise RuntimeError(
        f"""
    did not finish
    ntry      : {i}
    idx       : {idx}
    idx_nebr  : {idx_nebr}
    left lnz   : {left._get_lnz()}
    right lnz  : {right._get_lnz()}
    doneleft  : {doneLeft}
    doneright : {doneRight}
    """
    )


def _solve_spinodal(
    a, b, build_phases, idx, idx_nebr=None, efac=1.0, ref=None, build_kws=None, **kwargs
):
    if build_kws is None:
        build_kws = {}

    def f(x):
        c = build_phases(x, ref=ref, **build_kws)
        dw = c.wfe_phases.get_dw(idx, idx_nebr)
        out = dw - efac

        f._lnpi = c
        f._out = out

        return out

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)
    lnz = f._lnpi._get_lnz(build_phases.index)
    return lnz, r, f._lnpi


def _get_step(C, idx, idx_nebr):
    """
    find step value on

    if DeltabetaE[-1,ID] - DeltabetaE[0,ID]<0 -> step=+1 (go right)

    else step = -1
    """
    delta = C.zloc[[-1]].wfe_phases.get_dw(idx, idx_nebr) - C.zloc[
        [0]
    ].wfe_phases.get_dw(idx, idx_nebr)
    if delta == 0:
        raise ValueError("could not determine step, delta==0")
    elif delta < 0.0:
        return +1
    else:
        return -1


def get_spinodal(
    C,
    build_phases,
    idx,
    idx_nebr=None,
    efac=1.0,
    dlnz=0.5,
    dfac=1.0,
    vmin=0.0,
    vmax=1e5,
    ntry=20,
    step=None,
    nmax=20,
    ref=None,
    build_kws=None,
    close_kws=None,
    solve_kws=None,
    full_output=False,
):
    """
        locate spinodal point for a given phaseID
    _
        Parameters
        ----------
        ref : lnPiMasked
        C : lnPi_collection
            initial estimates to work from.  Function assumes C is in lnz sorted order
        idx, idx_nebr : int
            from/to phase id
        lnz_in : list
            list with value of static chem pot, and None for variable. e.g.,
            lnz_in=[None,0.0] implies lnz[0] is variable, and lnz[1]=0.0
        efac : float, optional
            cutoff value for spinodal
        dlnz : float, optional
            factor to kick back if C doesn't already have left and right bounds
        vmin : float, optional
            value denoting vmin, i.e., value of DeltabetaE if phaseID does not exist
        vmax : float, optional
            value indicating no transition, but phaseID==ID present
        ntry : int (Default 20)
            number of times to try kicking forward/backwards to find bracket
        step : int or None (Default None)
            if +1, step forward
            if -1, step backward
            if None, try to determine step
        nmax : int (Default 20)
            max number of steps to refine bracket
        build_phases : callable, optional
            function to create Phases.  Default is that fro get_default_PhaseCreator
        build_kws : dict, optional
            extra arguments to `build_phases`
        close_kws : dict, optional
            arguments to np.allclose
        solve_kws : dict, optional
            extra arguments to scipy.optimize.brentq
        full_output : bool (Default False)
            if true, return output info object

        Returns
        -------
        out : lnPi_phases object at spinodal point
        r :output info object (optional, returned if full_output is True)

    """
    assert len(C) > 1
    if build_kws is None:
        build_kws = {}
    if close_kws is None:
        close_kws = {}
    if solve_kws is None:
        solve_kws = {}

    if step is None:
        step = _get_step(C, idx=idx, idx_nebr=idx_nebr)

    assert step in [-1, +1]

    # if step == +1:
    #     CC = C
    # elif step == -1:
    #     CC = C[-1::-1]
    # else:
    #     raise ValueError('bad step')

    # get initial bracket
    L, R = _initial_bracket_spinodal_right(
        C,
        idx=idx,
        idx_nebr=idx_nebr,
        efac=efac,
        dlnz=dlnz,
        dfac=dfac,
        vmax=vmax,
        ntry=ntry,
        step=step,
        ref=ref,
        build_phases=build_phases,
        build_kws=build_kws,
    )

    left, right, rr = _refine_bracket_spinodal_right(
        L,
        R,
        idx=idx,
        idx_nebr=idx_nebr,
        efac=efac,
        nmax=nmax,
        vmin=vmin,
        vmax=vmax,
        ref=ref,
        build_phases=build_phases,
        build_kws=build_kws,
        close_kws=close_kws,
    )

    if left is None and right is None:
        # no spinodal found and left and right are close
        spin = None
        r = rr
    elif rr.converged:
        # converged to a solution
        spin = left
        r = rr
        r.bracket_iteration = rr.iterations
        r.from_solve = False
    else:
        # solve
        if step == -1:
            left, right = right, left

        a, b = [x._get_lnz(build_phases.index) for x in [left, right]]
        lnz, r, spin = _solve_spinodal(
            ref=ref,
            idx=idx,
            idx_nebr=idx_nebr,
            a=a,
            b=b,
            efac=efac,
            build_phases=build_phases,
            build_kws=build_kws,
            **solve_kws,
        )
        r.bracket_iterations = rr.iterations
        r.from_solve = True
    if full_output:
        return spin, r
    else:
        return spin


################################################################################
# Binodal routines


def get_binodal_point(
    IDs,
    lnz_min,
    lnz_max,
    build_phases,
    ref=None,
    build_kws=None,
    full_output=False,
    **kwargs,
):
    """
    calculate binodal point where Omega[ID[0]]==Omega[ID[1]]

    Parameters
    ----------
    ref : lnPi
        object to reweight
    IDs : tuple
        phase index of pair to equate
    lnz_min, lnz_max : float
        lnz_index values bracketing solution
    build_phases : callable
        function to create Phases object
    build_kws : dict, optional
        optional arguments to build_phases
    full_output : bool (Default False)
        if True, return solve stats
    kwargs : dict
        extra arguments to scipy.optimize.brentq
    Returns
    -------
    binodal : Phases object at binodal point
    stats : optional,
        solve stats object from brentq (optional, returned if full_output is True)
    """

    IDs = list(IDs)
    assert len(IDs) == 2

    if build_kws is None:
        build_kws = {}

    # lnz_idx = build_phases.index

    a, b = min(lnz_min, lnz_max), max(lnz_min, lnz_max)

    def f(x):
        p = build_phases(x, ref=ref, **build_kws)
        f.lnpi = p
        # Omegas = c.omega_phase()
        # return Omegas[IDs[0]] - Omegas[IDs[1]]
        return p.xge.betaOmega().reindex(phase=IDs).diff("phase").squeeze().values

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)
    r.residual = f(xx)
    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi


################################################################################
# Accessor classes/routines


class _BaseStability(object):
    """

    Parameters
    ----------
    collection: lnPiCollection
        Used to bracket the location limit of stability


    Methods
    -------
    __call__ : set value
    """

    _NAME = "base"

    def __init__(self, collection):
        self._parent = collection
        self.access_kws = {}

    @property
    def parent(self):
        return self._parent

    def set_values(self, items, info, index=None):
        self._items = items
        self._info = info
        self._index = index

    def set_access_kws(self, **kwargs):
        for k, v in kwargs.items():
            self.access_kws[k] = v

    @property
    def items(self):
        """Access to the underlying data"""
        return self._items

    def _get_access(self, items=None, concat_kws=None, **kwargs):
        if items is None:
            items = self._items
        if concat_kws is None:
            concat_kws = {}
        concat_kws = dict(names=[self._NAME], **concat_kws)
        kwargs = dict(self.access_kws, **kwargs)
        # return self._parent.concat_like(items, **concat_kws)
        # s = pd.concat({k:v._series for k,v in items}, **concat_kws)
        # return self._parent.new_like(s)
        return self._parent.concat(items, concat_kws=concat_kws, **kwargs)

    @gcached()
    def access(self):
        """lnPiCollection view of stability"""
        return self._get_access()

    def __getitem__(self, idx):
        return self._items[idx]

    def _get_appender(self, s=None):
        if s is None:
            s = self.access
        return s.droplevel(self._NAME)

    @property
    def appender(self):
        return self._get_appender()

    def append_to_parent(self, sort=True, copy_stability=True):
        new = self._parent.append(self.appender)
        if sort:
            new = new.sort_index()
        if copy_stability:
            setattr(new, self._NAME, self)
        return new


# NOTE : single create means this is only created once
@lnPiCollection.decorate_accessor("spinodal", single_create=False)
class Spinodals(_BaseStability):
    """
    Methods for calculation locations of spinodal
    """

    _NAME = "spinodal"

    def __call__(
        self,
        phase_ids,
        build_phases,
        efac=1.0,
        ref=None,
        build_kws=None,
        inplace=True,
        # append=False,
        force=False,
        as_dict=True,
        unstack=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        phase_ids : int, or sequence
            Phase ids to calculate spinodal for.  If int, then find spinodals for phases
            range(phase_ids).
        build_phases : callable
            Factory function to build phases.
            This should most likely be an instance of :class:`lnpy.BuildPhases_mu`
        efac : float, default=1.0
            Target value of `dw` to define spinodal.
        ref : lnPiMasked, optional
        build_kws : dict, optional
            optional arguments to `build_phases`
        inplace : bool, default=True
            if True, add spinodal inplace, otherwise return spinodal object
        force : bool, optional
            if True, force recalcuation of spinodal if alread set inplace.  Otherwise return already
            calculated values
        as_dict : bool, default=True
            if True, return dict of form {phase_id[0] : phases object, ...}.
        unstack : bool, optional
            if passed, create lnPiCollection objects with this unstack value.  If not passed,
            use unstack parameter from parent object
        **kwargs : extra argument to `get_spinodal` function

        Returns
        -------
        out : output
            if inplace, return self.
            if not inplace, and as dict, return dict, else return :class:`lnpy.lnPiCollection` with phase_id in index
        """

        if hasattr(self, "_items") and not force:
            if inplace:
                return self
            else:
                if as_dict:
                    out = self._items
                else:
                    out = self.access
                return out, self._info

        from .segment import BuildPhasesBase

        if not isinstance(build_phases, BuildPhasesBase):
            raise ValueError(
                "`build_phases` should be an instance of `BuildPhasesBase`."
                "Its likely an instance of `PhaseCreator.builphases`."
                "Instead, use an instance of `PhaseCreator.buildphases_mu`."
            )

        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError("phase_ids must be an int or list")

        out = {}
        info = {}
        kwargs["full_output"] = True
        for idx in phase_ids:
            s, r = get_spinodal(
                ref=ref,
                C=self._parent,
                idx=idx,
                build_phases=build_phases,
                build_kws=build_kws,
                **kwargs,
            )
            out[idx] = s
            info[idx] = r

        if unstack is None:
            unstack = self._parent._xarray_unstack
        self.set_access_kws(unstack=unstack)

        if inplace:
            self._items = out
            self._info = info
            # if append:
            #     return self._append_to_parent()
            # else:
            return self
        else:
            if not as_dict:
                out = self._get_access(out)
            return out, info


@lnPiCollection.decorate_accessor("binodal", single_create=False)
class Binodals(_BaseStability):
    """Routines to calculate binodal."""

    _NAME = "binodal"

    def get_pair(
        self,
        ids,
        lnz_min=None,
        lnz_max=None,
        spinodals=None,
        ref=None,
        build_phases=None,
        build_kws=None,
        **kwargs,
    ):

        if None in [lnz_min, lnz_max] and spinodals is None:
            spinodals = self._parent.spinodal
        if lnz_min is None:
            lnz_min = spinodals[ids[0]]._get_lnz(build_phases.index)
        if lnz_max is None:
            lnz_max = spinodals[ids[1]]._get_lnz(build_phases.index)
        return get_binodal_point(
            ref=ref,
            IDs=ids,
            lnz_min=lnz_min,
            lnz_max=lnz_max,
            build_phases=build_phases,
            build_kws=build_kws,
            **kwargs,
        )

    def __call__(
        self,
        phase_ids,
        build_phases,
        spinodals=None,
        ref=None,
        build_kws=None,
        inplace=True,
        force=False,
        as_dict=True,
        unstack=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        phase_ids : int, or sequence
            Phase ids to calculate binodal for.  If int, then find binodal for phases
            range(phase_ids).
        build_phases : callable
            Factory function to build phases
        spinodals : optional
            if not passes, then use parent.spinodal
            Used for bounding binodal
        ref : lnPiMasked, optional
        build_kws : dict, optional
            optional arguments to `build_phases`
        inplace : bool, default=True
            if True, add spinodal inplace, otherwise return spinodal object
        force : bool, optional
            if True, force recalcuation of spinodal if alread set inplace.  Otherwise return already
            calculated values
        as_dict : bool, default=True
            if True, return dict of form {phase_id[0] : phases object, ...}
        unstack : bool, optional
            if passed, create lnPiCollection objects with this unstack value.  If not passed,
            use unstack parameter from parent object
        **kwargs : extra argument to `get_spinodal` function

        Returns
        -------
        out : output
            if inplace, return self
            if not inplace, and as dict, return dict, else return lnPiCollection with phase_id in index
        """

        if inplace and hasattr(self, "_items") and not force:
            if inplace:
                # if append:
                #     if append_kws is None:
                #         append_kws = {}
                #     return self.append_to_parent(**append_kws)
                # else:
                return self
            else:
                if as_dict:
                    out = self._items
                else:
                    out = self.access
                return out, self._info

        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError("phase_ids must be an int or list")

        out = {}
        info = {}
        index = {}
        kwargs["full_output"] = True
        for idx, ids in enumerate(itertools.combinations(phase_ids, 2)):
            s, r = self.get_pair(
                ref=ref,
                ids=ids,
                spinodals=spinodals,
                build_phases=build_phases,
                build_kws=build_kws,
                **kwargs,
            )
            out[idx] = s
            info[idx] = r
            index[idx] = ids

        if unstack is None:
            unstack = self._parent._xarray_unstack
        self.set_access_kws(unstack=unstack)

        # either build output or inplace
        if inplace:
            self._items = out
            self._info = info
            self._index = index
            # if append:
            #     return self.append_to_parent()
            # else:
            return self
        else:
            if not as_dict:
                out = self._get_access(out)
            return out, info


@lnPiCollection.decorate_accessor("stability_append", single_create=False)
def _stability_append(self):
    """
    add stability from collection to this collection

    Parameters
    ----------
    other : optional, default=self
        if passed, copy stability from this collection to self, otherwise
        use self
    append: bool, default=True
        if True, append results to new frame
    sort: bool, default=True
        if True, sort appended results
    copy_stability
    """
    # Duplicate docstring so show up in docs

    def func(other=None, append=True, sort=True, copy_stability=True):
        """
        add stability from collection to this collection

        Parameters
        ----------
        other : optional, default=self
            if passed, copy stability from this collection to self, otherwise
            use self
        append: bool, default=True
            if True, append results to new frame
        sort: bool, default=True
            if True, sort appended results
        copy_stability
        """

        if (not append) and (not copy_stability):
            raise ValueError("one of append or copy_stability must be True")

        if other is None:
            other = self
        spin = other.spinodal
        bino = other.binodal
        if append:
            new = self.append(spin.appender).append(bino.appender)
            if sort:
                new = new.sort_index()
        else:
            new = self.copy()
        if copy_stability:
            new.spinodal = spin
            new.binodal = bino
        return new

    return func
