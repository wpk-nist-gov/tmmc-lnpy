"""
Thermodynamic stability (:mod:`~lnpy.stability`)
================================================

Calculation of spinodal and binodal
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, overload

import numpy as np
from module_utilities import cached

from .utils import RootResultDict, rootresults_to_rootresultdict

if TYPE_CHECKING:
    from typing import Any, Iterable, Literal, Mapping

    from scipy.optimize import RootResults
    from typing_extensions import Self

    from ._typing import MyNDArray
    from .lnpidata import lnPiMasked
    from .lnpiseries import lnPiCollection
    from .segment import BuildPhasesBase


class RootResultTotal(RootResultDict, total=False):
    """Modified root result"""

    left: lnPiCollection
    right: lnPiCollection
    left_done: bool
    right_done: bool
    info: str
    bracket_iteration: int
    from_solve: bool


def _rootresults_to_rootresulttotal(
    r: RootResults,
    *,
    left: lnPiCollection | None = None,
    right: lnPiCollection | None = None,
    left_done: bool | None = None,
    right_done: bool | None = None,
    info: str | None = None,
    bracket_iteration: int | None = None,
    from_solve: bool | None = None,
    residual: float | None = None,
) -> RootResultTotal:
    output = RootResultTotal(**rootresults_to_rootresultdict(r, residual=residual))  # type: ignore[typeddict-item]

    if left is not None:
        output["left"] = left
    if right is not None:
        output["right"] = right
    if left_done is not None:
        output["left_done"] = left_done
    if right_done is not None:
        output["right_done"] = right_done
    if info is not None:
        output["info"] = info
    if bracket_iteration is not None:
        output["bracket_iteration"] = bracket_iteration
    if from_solve is not None:
        output["from_solve"] = from_solve

    return output


# ###############################################################################
# Spinodal routines
def _initial_bracket_spinodal_right(  # noqa: C901,PLR0912
    collection: lnPiCollection,
    build_phases: BuildPhasesBase,
    idx: int,
    idx_nebr: int | None = None,
    efac: float = 1.0,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    # vmax: float = 1e5,
    ntry: int = 20,
    step: int = +1,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
) -> tuple[lnPiCollection, lnPiCollection]:
    """
    Find initial bracketing lnPiCollection for spinodal.

    Parameters
    ----------
    ref : lnPiMasked, optional
    build_phases : callable
        scalar function to build phases
    collection : lnPiColletion
        initial estimates to work from
    idx, idx_nebr : int
        id's of from/to phases.
    lnz_in : list
        list with value of static chem pot, and None for variable. e.g.,
        lnz_in=[None,0.0] implies lnz[0] is variable, and lnz[1]=0.0
    efac : float, default=1.0
        cutoff value for spinodal
    dlnz : float, default=0.5
        factor to kick back if collection doesn't already have left and right bounds
    vmax : float default=1e20
        value indicating no transition, but phase `idx` present.
    ntry : int, default=20
        number of times to try kicking forward/backwards to find bracket
    step : int, default=+1
        if +1, step forward
        if -1, step backward
    build_phases : callable, optional.
        function that returns a Phases object.
    build_kws : dict
        extra arguments to build_phases

    Returns
    -------
    left,right: lnPiCollection
        left and right bracketing :class:`~lnpy.lnpiseries.lnPiCollection` objects

    """

    if build_kws is None:
        build_kws = {}
    if efac <= 0:
        msg = "efac must be positive"
        raise ValueError(msg)

    # Series representation of dw
    s = collection.wfe.get_dw(idx, idx_nebr)
    if step < 0:
        s = s.iloc[-1::-1]

    # see if contain "left" bounds
    left = None
    s_idx = s[s > 0.0]
    if len(s_idx) == 0:
        msg = f"no phase {idx}"
        raise ValueError(msg)

    ss = s_idx[s_idx > efac]
    if len(ss) > 0:
        # get last one
        left = collection.mloc[ss.index[[-1]]]
    else:
        new_lnz = collection.mloc[s_idx.index[[0]]]._get_lnz(build_phases.index)
        for _i in range(ntry):
            new_lnz -= step * dlnz
            t = build_phases(new_lnz, ref=ref, **build_kws)
            if (
                idx in t._get_level("phase")
                and t.wfe_phases.get_dw(idx, idx_nebr) > efac
            ):
                left = t
                break

    if left is None:
        msg = "could not find left"
        raise RuntimeError(msg)

    # right
    right = None
    ss = s[s < efac]
    if len(ss) > 0:
        right = collection.mloc[ss.index[[0]]]
    else:
        new_lnz = collection.mloc[s.index[[-1]]]._get_lnz(build_phases.index)
        dlnz_ = dlnz
        for _i in range(ntry):
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
        msg = "could not find right"
        raise RuntimeError(msg)
    return left, right


def _refine_bracket_spinodal_right(
    left: lnPiCollection,
    right: lnPiCollection,
    build_phases: BuildPhasesBase,
    idx: int,
    idx_nebr: int | None = None,
    efac: float = 1.0,
    nmax: int = 30,
    vmax: float = 1e5,
    vmin: float = 0.0,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
    close_kws: Mapping[str, Any] | None = None,
) -> tuple[lnPiCollection | None, lnPiCollection | None, RootResultTotal]:
    """
    Find refined bracket.

    Finds where ``efac<wfe<vmax`` and
    ``vmin<wfe<efac``

    Parameters
    ----------
    left, right : lnPiCollection
        left and right initial brackets
    idx, idx_nebr : int
        from/to phase id's
    efac : float, default=1.0
        cutoff value for spinodal
    nmax : int, default=30
        max number of iterations
    vmin, vmax : float
        See above
    build_phases : callable
    build_kws : dict
    close_kwargs : dict
        arguments to :func:`numpy.allclose`

    Returns
    -------
    left,right : lnPiCollection
        left and right phases bracketing spinodal

    r : :class:`scipy.optimize.zeros.RootResults` object
    """
    from scipy.optimize import RootResults

    left_done = False
    right_done = False
    if build_kws is None:
        build_kws = {}
    if close_kws is None:
        close_kws = {}

    for i in range(nmax):
        # if idx in left.index and idx_nebr in left.index:
        # dw = _get_dw(left, idx, idx_nebr)
        dw = left.wfe_phases.get_dw(idx, idx_nebr)
        if dw < vmax and dw > efac:
            left_done = True

        # dw = _get_dw(right, idx, idx_nebr)
        dw = right.wfe_phases.get_dw(idx, idx_nebr)
        if dw > vmin and dw < efac:
            right_done = True

        #########
        # checks
        if left_done and right_done:
            # find bracket
            r = RootResults(root=None, iterations=i, function_calls=i, flag=1)
            return left, right, _rootresults_to_rootresulttotal(r)

        ########
        # converged?
        if np.allclose(left._get_lnz(), right._get_lnz(), **close_kws):
            # we've reached a breaking point
            if left_done:
                # can't find a lower bound to efac, just return where we're at
                r = RootResults(
                    root=left._get_lnz(), iterations=i + 1, function_calls=i, flag=0
                )

                r = _rootresults_to_rootresulttotal(
                    r,
                    left=left,
                    right=right,
                    left_done=left_done,
                    right_done=right_done,
                    info="all close and left_done",
                )
                return left, right, r

            # all close, and no good on either end -> no spinodal
            r = RootResults(root=None, iterations=i + 1, function_calls=i, flag=1)

            r = _rootresults_to_rootresulttotal(
                r,
                left=left,
                right=right,
                left_done=left_done,
                right_done=right_done,
                info="all clase and not left_done",
            )

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

    msg = f"""
    did not finish
    ntry      : {i}
    idx       : {idx}
    idx_nebr  : {idx_nebr}
    left lnz   : {left._get_lnz()}
    right lnz  : {right._get_lnz()}
    doneleft  : {left_done}
    doneright : {right_done}
    """
    raise RuntimeError(msg)


def _get_step(collection: lnPiCollection, idx: int, idx_nebr: int | None) -> int:
    """
    Find step value on

    if DeltabetaE[-1,ID] - DeltabetaE[0,ID]<0 -> step=+1 (go right)

    else step = -1
    """
    delta = collection.zloc[[-1]].wfe_phases.get_dw(idx, idx_nebr) - collection.zloc[
        [0]
    ].wfe_phases.get_dw(idx, idx_nebr)
    if delta == 0:
        msg = "could not determine step, delta==0"
        raise ValueError(msg)
    if delta < 0.0:
        return +1
    return -1


class _SolveSpinodal:
    collection: lnPiCollection

    def __init__(
        self,
        build_phases: BuildPhasesBase,
        idx: int,
        idx_nebr: int | None = None,
        efac: float = 1.0,
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.build_phases = build_phases
        self.idx = idx
        self.idx_nebr = idx_nebr
        self.efac = efac
        self.ref = ref
        self.build_kws = build_kws or {}

    def objective(self, x: float) -> float | MyNDArray:
        self.collection = self.build_phases(x, ref=self.ref, **self.build_kws)
        dw = self.collection.wfe_phases.get_dw(self.idx, self.idx_nebr)
        return dw - self.efac

    def solve(
        self, a: float, b: float, **kws: Any
    ) -> tuple[float, RootResultTotal, lnPiCollection]:
        from scipy.optimize import brentq

        xx, r = brentq(self.objective, a, b, full_output=True, **kws)

        return (
            self.collection._get_lnz(self.build_phases.index),
            _rootresults_to_rootresulttotal(r, residual=self.objective(xx)),  # type: ignore[arg-type]
            self.collection,
        )


def get_spinodal(  # noqa: PLR0913,PLR0917
    collection: lnPiCollection,
    build_phases: BuildPhasesBase,
    idx: int,
    idx_nebr: int | None = None,
    efac: float = 1.0,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    vmin: float = 0.0,
    vmax: float = 1e5,
    ntry: int = 20,
    step: int | None = None,
    nmax: int = 20,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
    close_kws: Mapping[str, Any] | None = None,
    solve_kws: Mapping[str, Any] | None = None,
) -> tuple[lnPiCollection | None, RootResultTotal]:
    """
    Locate spinodal point for a given pair of phases.

    Parameters
    ----------
    ref : lnPiMasked
    collection : lnPiCollection
        initial estimates to work from.  Function assumes collection is in lnz sorted order
    idx, idx_nebr : int
        from/to phase id
    lnz_in : list
        list with value of static chem pot, and None for variable. e.g.,
        lnz_in=[None,0.0] implies lnz[0] is variable, and lnz[1]=0.0
    efac : float, optional
        cutoff value for spinodal
    dlnz : float, optional
        factor to kick back if collection doesn't already have left and right bounds
    vmin : float, optional
        Value denoting ``vmin``, i.e., value of free energy difference phase does not exist.
    vmax : float, optional
        value indicating no transition, but phase `idx` is present.
    ntry : int, default=20
        number of times to try kicking forward/backwards to find bracket
    step : int or None, default=None
        if +1, step forward
        if -1, step backward
        if None, try to determine step
    nmax : int, default=20
        max number of steps to refine bracket
    build_phases : callable, optional
        Function to create Phases.  Default is that from get_default_phasecreator
    build_kws : dict, optional
        extra arguments to ``build_phases``
    close_kws : dict, optional
        arguments to :func:`numpy.allclose`
    solve_kws : dict, optional
        extra arguments to :func:`scipy.optimize.brentq`

    Returns
    -------
    out : :class:`~lnpy.lnpiseries.lnPiCollection` object at spinodal point
    r : object
        Info object

    """
    assert len(collection) > 1
    build_kws = build_kws or {}
    close_kws = close_kws or {}
    solve_kws = solve_kws or {}

    step = step or _get_step(collection, idx=idx, idx_nebr=idx_nebr)

    assert step in {-1, +1}

    # get initial bracket
    left_initial, right_initial = _initial_bracket_spinodal_right(
        collection,
        idx=idx,
        idx_nebr=idx_nebr,
        efac=efac,
        dlnz=dlnz,
        dfac=dfac,
        # vmax=vmax,
        ntry=ntry,
        step=step,
        ref=ref,
        build_phases=build_phases,
        build_kws=build_kws,
    )

    left, right, rr = _refine_bracket_spinodal_right(
        left_initial,
        right_initial,
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
    elif rr["converged"]:
        # converged to a solution
        spin = left
        r = rr
        r["bracket_iteration"] = rr["iterations"]
        r["from_solve"] = False
    else:
        assert left is not None
        assert right is not None
        # solve
        if step == -1:
            left, right = right, left

        a, b = (x._get_lnz(build_phases.index) for x in [left, right])
        _lnz, r, spin = _SolveSpinodal(
            ref=ref,
            idx=idx,
            idx_nebr=idx_nebr,
            efac=efac,
            build_phases=build_phases,
            build_kws=build_kws,
        ).solve(a=a, b=b, **solve_kws)

        r["bracket_iteration"] = rr["iterations"]
        r["from_solve"] = True

    return spin, r


################################################################################
# Binodal routines
class _SolveBinodal:
    """
    Class to calculate binodal

    Parameters
    ----------
    ref : lnPiMasked
        object to reweight
    build_phases : callable
        function to create Phases object
    build_kws : dict, optional
        optional arguments to build_phases
    """

    collection: lnPiCollection

    def __init__(
        self,
        build_phases: BuildPhasesBase,
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.build_phases = build_phases
        self.ref = ref
        self.build_kws = build_kws or {}

    def objective(self, x: float) -> float | MyNDArray:
        self.collection = self.build_phases(x, ref=self.ref, **self.build_kws)
        return (
            self.collection.xge.betaOmega()
            .reindex(phase=self.ids)
            .diff("phase")
            .squeeze()
            .values
        )

    def solve(
        self, ids: tuple[int, int], lnz_min: float, lnz_max: float, **kws: Any
    ) -> tuple[lnPiCollection, RootResultTotal]:
        """
        Calculate binodal point where Omega[ID[0]]==Omega[ID[1]]

        Parameters
        ----------
        ids : tuple
            phase index of pair to equate
        lnz_min, lnz_max : float
            lnz_index values bracketing solution
        **kwargs :
            extra arguments to ``brentq``

        Returns
        -------
        binodal : lnPiCollection
            At binodal point.
        stats : object, optional
            Solve stats object from ``brentq`` (optional, returned if full_output is True)

        See Also
        --------
        scipy.optimize.brentq
        """

        from scipy.optimize import brentq

        assert len(ids) == 2
        self.ids = ids

        a, b = min(lnz_min, lnz_max), max(lnz_min, lnz_max)

        xx, r = brentq(self.objective, a, b, full_output=True, **kws)

        return self.collection, _rootresults_to_rootresulttotal(
            r,
            residual=self.objective(xx),  # type: ignore[arg-type]
        )


################################################################################
# Accessor classes/routines
class StabilityBase:
    """
    Base class for stability

    Parameters
    ----------
    collection: lnPiCollection
        Used to bracket the location limit of stability

    """

    _NAME = "base"

    _items: dict[int, lnPiCollection | None]
    _info: dict[int, RootResultTotal]

    def __init__(self, collection: lnPiCollection) -> None:
        self._parent = collection
        self.access_kws: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}

    @property
    def parent(self) -> lnPiCollection:
        """Accessor to parent :class:`~lnpy.lnpiseries.lnPiCollection` object"""
        return self._parent

    def set_access_kws(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.access_kws[k] = v

    @property
    def items(self) -> dict[int, lnPiCollection | None]:
        """Access to the underlying data"""
        return self._items

    def _get_access(
        self,
        items: Mapping[int, lnPiCollection | None] | None = None,
        concat_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> lnPiCollection:
        items = items or self._items
        concat_kws = concat_kws or {}

        concat_kws = dict(names=[self._NAME], **concat_kws)
        kwargs = dict(self.access_kws, **kwargs)
        # return self._parent.concat_like(items, **concat_kws)
        # s = pd.concat({k:v._series for k,v in items}, **concat_kws)
        # return self._parent.new_like(s)

        return self._parent.concat(items, concat_kws=concat_kws, **kwargs)  # type: ignore[arg-type]

    @cached.prop
    def access(self) -> lnPiCollection:
        """View (:class:`lnpy.lnpiseries.lnPiCollection`) of stability"""
        return self._get_access()

    def __getitem__(self, idx: int) -> lnPiCollection | None:
        return self._items[idx]

    def _get_appender(self, s: lnPiCollection | None = None) -> lnPiCollection:
        if s is None:
            s = self.access
        return s.droplevel(self._NAME)

    @property
    def appender(self) -> lnPiCollection:
        """View of :attr:`access` to be appended to :attr:`parent`"""
        return self._get_appender()

    def append_to_parent(
        self, sort: bool = True, copy_stability: bool = True
    ) -> lnPiCollection:
        """New collection with spinodal data appended to parent."""
        new = self._parent.append(self.appender)
        if sort:
            new = new.sort_index()
        if copy_stability:
            # workaround to copy over spinodal
            new._cache["spinodal"] = self
        return new


# NOTE : single create means this is only created once
# @lnPiCollection.decorate_accessor("spinodal", single_create=False)
class Spinodals(StabilityBase):
    """Methods for calculation locations of spinodal"""

    _NAME = "spinodal"

    _items: dict[int, lnPiCollection | None]
    _info: dict[int, RootResultTotal]

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: Literal[True] = ...,
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: Literal[False],
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> tuple[
        dict[int, lnPiCollection | None] | lnPiCollection, dict[int, RootResultTotal]
    ]:
        ...

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: bool,
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> (
        Self
        | tuple[
            dict[int, lnPiCollection | None] | lnPiCollection,
            dict[int, RootResultTotal],
        ]
    ):
        ...

    def __call__(  # noqa: C901
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: bool = True,
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
        force: bool = False,
        as_dict: bool = True,
        unstack: bool | None = None,
        raise_unconverged: bool = True,
        **kwargs: Any,
    ) -> (
        Self
        | tuple[
            dict[int, lnPiCollection | None] | lnPiCollection,
            dict[int, RootResultTotal],
        ]
    ):
        """
        Parameters
        ----------
        phase_ids : int, or sequence
            Phase ids to calculate spinodal for.  If int, then find spinodals for phases
            range(phase_ids).
        build_phases : callable
            Factory function to build phases.
            This should most likely be an instance of :class:`lnpy.segment.BuildPhasesBase`
        ref : lnPiMasked, optional
        build_kws : dict, optional
            optional arguments to `build_phases`
        inplace : bool, default=True
            if True, add spinodal inplace, otherwise return spinodal object
        force : bool, optional
            if True, force recalculation of spinodal if already set inplace.  Otherwise return already
            calculated values
        as_dict : bool, default=True
            if True, return dict of form {phase_id[0] : phases object, ...}.
        unstack : bool, optional
            if passed, create lnPiCollection objects with this unstack value.  If not passed,
            use unstack parameter from parent object
        raise_unconverged : bool, default=True
            If True, raise error if calculation does not converge.
        **kwargs :
            Extra argument to :meth:`get_spinodal` function

        Returns
        -------
        out : dict or lnPiCollection, optional
            if inplace, return self.
            if not inplace, and as dict, return dict, else return :class:`lnpy.lnpiseries.lnPiCollection` with phase_id in index
        """

        if hasattr(self, "_items") and not force:
            if inplace:
                return self
            if as_dict:
                return self._items, self._info
            return self.access, self._info

        from .segment import BuildPhasesBase

        if not isinstance(build_phases, BuildPhasesBase):
            msg = (
                "`build_phases` should be an instance of `BuildPhasesBase`."
                "Its likely an instance of `PhaseCreator.builphases`."
                "Instead, use an instance of `PhaseCreator.buildphases_mu`."
            )
            raise ValueError(msg)

        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        else:
            phase_ids = list(phase_ids)

        out: dict[int, lnPiCollection | None] = {}
        info: dict[int, RootResultTotal] = {}
        for idx in phase_ids:
            s, r = get_spinodal(
                ref=ref,
                collection=self._parent,
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

        converged = all(x["converged"] for x in info.values())
        if raise_unconverged and not converged:
            msg = "Spinodal calculation did not converge"
            raise ValueError(msg)

        if inplace:
            self._items = out
            self._info = info
            # if append:
            #     return self._append_to_parent()
            # else:
            return self

        # total convergence:
        if not as_dict and converged:
            return self._get_access(out), info
        return out, info


# @lnPiCollection.decorate_accessor("binodal", single_create=False)
class Binodals(StabilityBase):
    """Routines to calculate binodal."""

    _NAME = "binodal"
    _items: dict[int, lnPiCollection]  # type: ignore[assignment]

    def get_pair(
        self,
        ids: tuple[int, int],
        solver: _SolveBinodal,
        lnz_min: float | None = None,
        lnz_max: float | None = None,
        spinodals: Spinodals | None = None,
        **kwargs: Any,
    ) -> tuple[lnPiCollection, RootResultTotal]:
        if None in {lnz_min, lnz_max} and spinodals is None:
            spinodals = self._parent.spinodal

        def _get_lnz(idx: int) -> float:
            assert spinodals is not None
            s = spinodals[idx]
            if s is None:
                msg = f"spinodal with index={idx} is None"
                raise ValueError(msg)
            return s._get_lnz(solver.build_phases.index)

        lnz_min = lnz_min or _get_lnz(ids[0])
        lnz_max = lnz_max or _get_lnz(ids[1])

        return solver.solve(ids=ids, lnz_min=lnz_min, lnz_max=lnz_max, **kwargs)

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: Literal[True] = ...,
        spinodals: Spinodals | None = ...,
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: Literal[False],
        spinodals: Spinodals | None = ...,
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> tuple[dict[int, lnPiCollection] | lnPiCollection, dict[int, RootResultTotal]]:
        ...

    @overload
    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: bool,
        spinodals: Spinodals | None = ...,
        ref: lnPiMasked | None = ...,
        build_kws: Mapping[str, Any] | None = ...,
        force: bool = ...,
        as_dict: bool = ...,
        unstack: bool | None = ...,
        raise_unconverged: bool = ...,
        **kwargs: Any,
    ) -> (
        Self
        | tuple[dict[int, lnPiCollection] | lnPiCollection, dict[int, RootResultTotal]]
    ):
        ...

    def __call__(
        self,
        phase_ids: int | Iterable[int],
        build_phases: BuildPhasesBase,
        *,
        inplace: bool = True,
        spinodals: Spinodals | None = None,
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
        force: bool = False,
        as_dict: bool = True,
        unstack: bool | None = None,
        raise_unconverged: bool = True,
        **kwargs: Any,
    ) -> (
        Self
        | tuple[dict[int, lnPiCollection] | lnPiCollection, dict[int, RootResultTotal]]
    ):
        """
        Parameters
        ----------
        phase_ids : int, or sequence
            Phase ids to calculate binodal for.  If int, then find binodal for phases
            range(phase_ids).
        build_phases : callable
            Factory function to build phases
        inplace : bool, default=True
            if True, add binodal inplace, otherwise return spinodal object
        spinodals : optional
            if not passes, then use parent.spinodal
            Used for bounding binodal
        ref : lnPiMasked, optional
        build_kws : dict, optional
            optional arguments to `build_phases`
        force : bool, optional
            if True, force recalculation of spinodal if already set inplace.  Otherwise return already
            calculated values
        as_dict : bool, default=True
            if True, return dict of form {phase_id[0] : phases object, ...}
        unstack : bool, optional
            if passed, create lnPiCollection objects with this unstack value.  If not passed,
            use unstack parameter from parent object
        raise_unconverged : bool, default=True
            If True, raise error if calculation does not converge.
        **kwargs
            Extra argument to :meth:`get_spinodal` function

        Returns
        -------
        out : object
            if inplace, return self
            if not inplace, and as dict, return dict, else return lnPiCollection with phase_id in index
        """

        if inplace and hasattr(self, "_items") and not force:
            if inplace:
                return self

            out = self._items if as_dict else self.access
            return out, self._info

        self._solver = _SolveBinodal(
            ref=ref, build_phases=build_phases, build_kws=build_kws
        )

        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        else:
            phase_ids = list(phase_ids)

        out = {}
        info = {}
        index = {}
        for idx, ids in enumerate(itertools.combinations(phase_ids, 2)):
            s, r = self.get_pair(
                ids=ids,
                spinodals=spinodals,
                solver=self._solver,
                **kwargs,
            )
            out[idx] = s
            info[idx] = r
            index[idx] = ids

        converged = all(x["converged"] for x in info.values())
        if raise_unconverged and not converged:
            msg = "Binodal calculation did not converge"
            raise ValueError(msg)

        if unstack is None:
            unstack = self._parent._xarray_unstack
        self.set_access_kws(unstack=unstack)

        # either build output or inplace
        if inplace:
            self._items = out
            self._info = info
            self._index = index
            return self

        if not as_dict and converged:
            return self._get_access(out), info
        return out, info


# @lnPiCollection.decorate_accessor("stability_append", single_create=False)
# def _stability_append(self):
#     """
#     Add stability from a collection to this collection

#     Parameters
#     ----------
#     other : optional, optional
#         if passed, copy stability from this collection to self, otherwise
#         use `self`
#     append: bool, default=True
#         if True, append results to new frame
#     sort: bool, default=True
#         if True, sort appended results
#     copy_stability
#     """
#     # Duplicate docstring so show up in docs

#     def func(other=None, append=True, sort=True, copy_stability=True):
#         """
#         Add stability from collection to this collection

#         Parameters
#         ----------
#         other : optional, optional
#             if passed, copy stability from this collection to self, otherwise
#             use `self`
#         append: bool, default=True
#             if True, append results to new frame
#         sort: bool, default=True
#             if True, sort appended results
#         copy_stability
#         """

#         if (not append) and (not copy_stability):
#             raise ValueError("one of append or copy_stability must be True")

#         if other is None:
#             other = self
#         spin = other.spinodal
#         bino = other.binodal
#         if append:
#             new = self.append(spin.appender).append(bino.appender)
#             if sort:
#                 new = new.sort_index()
#         else:
#             new = self.copy()
#         if copy_stability:
#             # TODO(wpk): fix this hack
#             new._cache["spinodal"] = spin
#             new._cache["binodal"] = bino
#             # new.spinodal = spin
#             # new.binodal = bino
#         return new

#     return func
