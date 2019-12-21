"""
thermodynamic stability routines (i.e., spinodal/binodal)
"""
import itertools

import numpy as np
from scipy import optimize

from .core import CollectionPhases
from .cached_decorators import gcached


# ###############################################################################
# Spinodal routines
def _initial_bracket_spinodal_right(C,
                                    build_phases,
                                    idx,
                                    idx_nebr=None,
                                    efac=1.0,
                                    dlnz=0.5,
                                    vmax=1e5,
                                    ntry=20,
                                    step=+1,
                                    ref=None,
                                    build_kws=None):
    """
    find initial bracketing lnpi_phases of phaseID==ID bracketing point where
    DeltabetaE_phaseIDS()[ID]==efac

    Parameters
    ----------
    ref : MaskedlnPi, optional
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

    # delta E
    dE = C.wlnPi.get_delta_w(idx, idx_nebr)

    # where idx exists
    has_idx = np.array([idx in x.index for x in C])

    # find locations where have 'ID'
    if has_idx.sum() == 0:
        raise ValueError('no phase %i' % idx)
    w = np.where(has_idx)[0]

    # left
    left = None
    for i in w[-1::-1]:
        if dE[i] > efac:
            left = C[i]
            break

    if left is None:
        # need to find a new value lnz bounding thing

        new_lnz = C[w[0]].lnz[build_phases.index]
        for i in range(ntry):
            new_lnz -= step * dlnz
            t = build_phases(new_lnz, ref=ref, **build_kws)
            if idx in t.index:
                # _get_dw(t, idx, idx_nebr)
                dw = t.wlnPi.get_delta_w(idx, idx_nebr)
                if dw > efac:
                    left = t
                    break
        if left is None:
            raise RuntimeError('could not find left')

    #right
    right = None
    for i in w:
        if (dE[i] < efac):
            right = C[i]
            break

    if right is None:
        if w[-1] + 1 < len(C):
            right = C[w[-1] + 1]
        else:
            new_lnz = C[w[-1]].lnz[build_phases.index]

            for i in range(ntry):
                new_lnz += step * dlnz
                t = build_phases(new_lnz, ref=ref, **build_kws)
                if idx not in t.index:
                    right = t
                    break
            if right is None:
                raise RuntimeError('could not find right')

    return left, right


def _refine_bracket_spinodal_right(left,
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
                                   close_kws=None):
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
        #if idx in left.index and idx_nebr in left.index:
        # dw = _get_dw(left, idx, idx_nebr)
        dw = left.wlnPi.get_delta_w(idx, idx_nebr)
        if dw < vmax and dw > efac:
            doneLeft = True

        # dw = _get_dw(right, idx, idx_nebr)
        dw = right.wlnPi.get_delta_w(idx, idx_nebr)
        if dw > vmin and dw < efac:
            doneRight = True

        #########
        #checks
        if doneLeft and doneRight:
            #find bracket
            r = optimize.zeros.RootResults(root=None,
                                           iterations=i,
                                           function_calls=i,
                                           flag=1)
            return left, right, r

        ########
        #converged?
        if np.allclose(left.lnz, right.lnz, **close_kws):
            #we've reached a breaking point
            if doneLeft:
                #can't find a lower bound to efac, just return where we're at
                r = optimize.zeros.RootResults(root=left.lnz,
                                               iterations=i + 1,
                                               function_calls=i,
                                               flag=0)
                for k, val in [('left', left), ('right', right),
                               ('doneleft', doneLeft),
                               ('doneright', 'doneRight'),
                               ('info', 'all close and doneleft')]:
                    setattr(r, k, val)
                return left, right, r

            #elif not doneLeft and not doneRight:
            else:
                #all close, and no good on either end -> no spinodal
                r = optimize.zeros.RootResults(root=None,
                                               iterations=i + 1,
                                               function_calls=i,
                                               flag=1)
                for k, val in [('left', left), ('right', right),
                               ('doneleft', doneLeft),
                               ('doneright', 'doneRight'),
                               ('info', 'all close and doneleft')]:
                    setattr(r, k, val)
                for k, val in [('left', left), ('right', right),
                               ('info', 'all close and not doneleft')]:
                    setattr(r, k, val)
                return None, None, r

        # mid point phases
        lnz_mid = 0.5 * (left.lnz[build_phases.index] +
                         right.lnz[build_phases.index])
        mid = build_phases(lnz_mid, ref=ref, **build_kws)
        dw = mid.wlnPi.get_delta_w(idx, idx_nebr)

        if idx in mid.index and dw >= efac:
            left = mid
        else:
            right = mid

    raise RuntimeError(f"""
    did not finish
    ntry      : {i}
    idx       : {idx}
    idx_nebr  : {idx_nebr}
    left lnz   : {left.lnz}
    right lnz  : {right.lnz}
    doneleft  : {doneleft}
    doneright : {doneright}
    """)


def _solve_spinodal(a,
                    b,
                    build_phases,
                    idx,
                    idx_nebr=None,
                    efac=1.0,
                    ref=None,
                    build_kws=None,
                    **kwargs):
    if build_kws is None:
        build_kws = {}

    def f(x):
        c = build_phases(x, ref=ref, **build_kws)
        dw = c.wlnPi.get_delta_w(idx, idx_nebr)

        out = dw - efac

        f._lnpi = c
        f._out = out

        return out

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)
    lnz = f._lnpi.lnz[build_phases.index]
    return lnz, r, f._lnpi


def _get_step(C, idx, idx_nebr):
    """
    find step value on

    if DeltabetaE[-1,ID] - DeltabetaE[0,ID]<0 -> step=+1 (go right)

    else step = -1
    """
    delta = (C[-1].wlnPi.get_delta_w(idx, idx_nebr) -
             C[0].wlnPi.get_delta_w(idx, idx_nebr))

    if delta == 0:
        raise ValueError('could not determine step, delta==0')
    elif delta < 0.0:
        return +1
    else:
        return -1


def get_spinodal(C,
                 build_phases,
                 idx,
                 idx_nebr=None,
                 efac=1.0,
                 dlnz=0.5,
                 vmin=0.0,
                 vmax=1e5,
                 ntry=20,
                 step=None,
                 nmax=20,
                 ref=None,
                 build_kws=None,
                 nphases_max=None,
                 close_kws=None,
                 solve_kws=None,
                 full_output=False):
    """
    locate spinodal point for a given phaseID

    Parameters
    ----------
    ref : MaskedlnPi
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
    nphases_max : int
        max number of phases.  To be used in get_default_PhaseCreator if passed `build_phases` is None
    close_kws : dict, optional
        arguments to np.allclose
    solve_kws : dict, optional
        extra arguments to scipy.optimize.brentq
    full_output : bool (Default False)
        if true, return output info object

    Returns
    -------
    out : lnPi_phases object at spinodal point
    r : output info object (optional, returned if full_output is True)

    """
    assert (len(C) > 1)
    if build_kws is None:
        build_kws = {}
    if close_kws is None:
        close_kws = {}
    if solve_kws is None:
        solve_kws = {}

    if step is None:
        step = _get_step(C, idx=idx, idx_nebr=idx_nebr)
    if step == +1:
        CC = C
    elif step == -1:
        CC = C[-1::-1]
    else:
        raise ValueError('bad step')

    msk = C[0].lnz != C[1].lnz
    assert msk.sum() == 1

    #get initial bracket
    L, R = _initial_bracket_spinodal_right(CC,
                                           idx=idx,
                                           idx_nebr=idx_nebr,
                                           efac=efac,
                                           dlnz=dlnz,
                                           vmax=vmax,
                                           ntry=ntry,
                                           step=step,
                                           ref=ref,
                                           build_phases=build_phases,
                                           build_kws=build_kws)

    left, right, rr = _refine_bracket_spinodal_right(L,
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
                                                     close_kws=close_kws)

    if left is None and right is None:
        #no spinodal found and left and right are close
        spin = None
        r = rr
    elif rr.converged:
        #converged to a solution
        spin = left
        r = rr
        r.bracket_iteration = rr.iterations
        r.from_solve = False
    else:
        #solve
        if step == -1:
            left, right = right, left
        a, b = left.lnz[build_phases.index], right.lnz[build_phases.index]
        lnz, r, spin = _solve_spinodal(ref=ref,
                                       idx=idx,
                                       idx_nebr=idx_nebr,
                                       a=a,
                                       b=b,
                                       efac=efac,
                                       build_phases=build_phases,
                                       build_kws=build_kws,
                                       **solve_kws)
        r.bracket_iterations = rr.iterations
        r.from_solve = True
    if full_output:
        return spin, r
    else:
        return spin


################################################################################
# Binodal routines


def get_binodal_point(IDs,
                      lnzA,
                      lnzB,
                      build_phases,
                      ref=None,
                      build_kws=None,
                      nphases_max=None,
                      full_output=False,
                      **kwargs):
    """
    calculate binodal point where Omega[ID[0]]==Omega[ID[1]]

    Parameters
    ----------
    ref : lnPi
        object to reweight
    IDs : tuple
        phase index of pair to equate
    lnzA, lnzB : float
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

    lnz_idx = build_phases.index

    a, b = min(lnzA, lnzB), max(lnzA, lnzB)

    def f(x):
        c = build_phases(x, ref=ref, **build_kws)
        f.lnpi = c
        # Omegas = c.omega_phase()
        # return Omegas[IDs[0]] - Omegas[IDs[1]]
        return c.xgce.betaOmega().reindex(phase=IDs).diff('phase')

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)
    r.residual = f(xx)
    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi


################################################################################
# Accessor classes/routines


class _BaseStability(object):
    _NAME = 'base'


    def __init__(self, collection):
        self._c = collection

    @property
    def items(self):
        return self._items

    @gcached()
    def access(self):
        index, items = zip(*[(k, v) for k, v in self.items.items()])
        # return CollectionPhases(items, index=index)
        # for time being don't pass index
        # this is a helper accessor anyway
        return self._c.new_like(items=items, index=index, concat_coords='all')

    # def __getattr__(self, attr):
    #     if hasattr(self.access, attr):
    #         return getattr(self.access,'attr')
    #     else:
    #         raise AttributeError('no attribute {}'.format(attr))

    def __getitem__(self, idx):
        return self._items[idx]

    def index_collection(self, dtype=np.uint8):
        """
        return an array of same length as parent collection
        with each stability kind marked by a value > 0
        if this value == 0, then no stabilty point
        if this value >  0, then value - 1 is the stability type index
        """
        out = []
        for rec, p in enumerate(self._c):
            val = 0
            for idx, s in self.items.items():
                if s is p:
                    val = idx + 1
                    break
            out.append(val)
        return np.array(out, dtype=dtype)

    def set_by_index(self, index):
        """
        set by index array
        """
        features = np.unique(index[index > 0])
        items = {}
        for feature in features:
            idx = np.where(index == feature)[0][0]
            items[feature - 1] = self._c[idx]
        self._items = items

    def assign_coords(self, da, name=None, dim=None, dtype=np.uint8):
        """
        add in index to dataarray
        """
        if name is None:
            name = self._NAME
        if dim is None:
            dim = self._concat_dim
        kws = {name: (dim, self.index_collection(dtype=dtype))}
        return (da.assign_coords(**kws))

    def from_dataarray(self, da, name=None):
        """set from dataarray"""
        if name is None:
            name = self._NAME
        self.set_by_index(da[name].values)


# NOTE : single create means this is only created once
@CollectionPhases.decorate_accessor('spinodals', single_create=True)
class Spinodals(_BaseStability):
    _NAME = 'spinodal'

    def __call__(self,
                 phase_ids,
                 build_phases,
                 ref=None,
                 build_kws=None,
                 nphases_max=None,
                 inplace=False,
                 append=False,
                 force=False,
                 as_dict=True,
                 **kwargs):

        if inplace and hasattr(self, '_items') and not force:
            raise ValueError('can reset inplace without force')
        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError('phase_ids must be an int or list')

        out = {}
        info = {}
        kwargs['full_output'] = True
        for idx in phase_ids:
            s, r = get_spinodal(ref=ref,
                                C=self._c,
                                idx=idx,
                                build_phases=build_phases,
                                build_kws=build_kws,
                                nphases_max=nphases_max,
                                **kwargs)
            out[idx] = s
            info[idx] = r

        if append:
            for v in out.values():
                self._c.append(v, index=None, get_index=False)
#            self._c.extend([v for v in out.values() if v is not None])
        if inplace:
            self._items = out
            self._info = info
        else:
            if as_dict:
                return out, info
            else:
                index, items = zip(*[(k, v) for k, v in out.items()])
                # return CollectionPhases(items, index=index)
                # for time being don't pass index
                # this is a helper accessor anyway
                return CollectionPhases(items, index=index), info


@CollectionPhases.decorate_accessor('binodals', single_create=True)
class Binodals(_BaseStability):
    _NAME = 'binodal'

    def get_pair(self,
                 ids,
                 lnzA=None,
                 lnzB=None,
                 spinodals=None,
                 ref=None,
                 build_phases=None,
                 build_kws=None,
                 nphases_max=None,
                 **kwargs):

        if None in [lnzA, lnzB] and spinodals is None:
            spinodals = self._c.spinodals
        if lnzA is None:
            lnzA = spinodals[ids[0]].lnz[build_phases.index]
        if lnzB is None:
            lnzB = spinodals[ids[1]].lnz[build_phases.index]
        return get_binodal_point(ref=ref,
                                 IDs=ids,
                                 lnzA=lnzA,
                                 lnzB=lnzB,
                                 build_phases=build_phases,
                                 build_kws=build_kws,
                                 nphases_max=nphases_max,
                                 **kwargs)

    def __call__(self,
                 phase_ids,
                 build_phases,
                 spinodals=None,
                 ref=None,
                 build_kws=None,
                 nphases_max=None,
                 inplace=False,
                 append=False,
                 force=False,
                 as_dict=True,
                 **kwargs):

        if inplace and not force and hasattr(self, '_items'):
            raise ValueError('can reset inplace without force')
        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError('phase_ids must be an int or list')

        out = {}
        info = {}
        index = {}
        kwargs['full_output'] = True
        for idx, ids in enumerate(itertools.combinations(phase_ids, 2)):
            s, r = self.get_pair(ref=ref,
                                 ids=ids,
                                 spinodals=spinodals,
                                 build_phases=build_phases,
                                 build_kws=build_kws,
                                 nphases_max=nphases_max,
                                 **kwargs)
            out[idx] = s
            info[idx] = r
            index[idx] = ids
        if append:
            for v in out.values():
                self._c.append(v, index=None, get_index=False)
#            self._c.extend([v for v in out.values() if v is not None])
        if inplace:
            self._items = out
            self._info = info
            self._index = index
        else:
            if as_dict:
                return out, info
            else:
                index, items = zip(*[(k, v) for k, v in out.items()])
                # return CollectionPhases(items, index=index)
                # for time being don't pass index
                # this is a helper accessor anyway
                return CollectionPhases(items, index=index), info
