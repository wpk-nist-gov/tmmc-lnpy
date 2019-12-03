import numpy as np
from scipy import optimize
from .segment import get_default_PhaseCreator



# def _get_dw(p, idx, idx_nebr=None):
#     # if idx_nebr is None, consider
#     # all neighbors and return min dw
#     has_idx = idx in p.index
#     if not has_idx:
#         return 0.0
#     if idx_nebr is None:
#         nebrs = p.index.drop(idx)
#     elif idx_nebr in p.index:
#         nebrs = [idx_nebr]
#     else:
#         nebrs = []

#     if len(nebrs) == 0:
#         return np.inf
#     dw = (
#         p.wlnPi.delta_w.sel(phase=idx, phase_nebr=nebrs)
#         .min('phase_nebr')
#         .values
#     )
#     return dw

def _initial_bracket_spinodal_right(C,
                                    lnz_in,
                                    idx,
                                    idx_nebr=None,
                                    efac=1.0,
                                    dlnz=0.5,
                                    vmax=1e5,
                                    ntry=20,
                                    step=+1,
                                    ref=None,
                                    build_phases=None,
                                    build_kws=None):
    """
    find initial bracketing lnpi_phases of phaseID==ID bracketing point where DeltabetaE_phaseIDS()[ID]==efac

    Parameters
    ----------
    ref : MaskedlnPi
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
    if build_phases is None:
        raise ValueError('must supply build_phases')

    # lnz which varies
    lnz_idx = lnz_in.index(None)
    # delta E
    # dE = C.wlnPi.delta_w.sel(
    #     phase=idx, phase_nebr=idx_nebr).fillna(np.inf).values
    # dE = np.array([_get_dw(p, idx, idx_nebr) for p in C])
    dE = C.wlnPi.get_delta_w(idx, idx_nebr)

    # where idx exists
    has_idx = np.array([idx in x.index for x in C])

    #find locations where have 'ID'
    if has_idx.sum() == 0:
        raise ValueError('no phase %i' % idx)
    w = np.where(has_idx)[0]

    #left
    left = None
    for i in w[-1::-1]:
        if dE[i] > efac and has_idx[i]:
            left = C[i]
            break

    if left is None:
        #need to find a new value lnz bounding thing
        new_lnz = lnz_in[:]
        new_lnz[lnz_idx] = C[w[0]].lnz[lnz_idx]
        new_lnz = np.asarray(newd_lnz)

        for i in range(ntry):
            new_lnz[lnz_idx] -= step * dlnz 
            t = build_phases(ref=ref, lnz=new_lnz, **build_kws)
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
    for i in w[-1::-1]:
        if (not has_idx[i]) or (dE[i] < efac):
            right = C[i]
            break

    if right is None:
        if w[-1] + 1 < len(C):
            right = C[w[-1] + 1]
        else:
            new_lnz = lnz_in[:]
            new_lnz[lnz_idx] = C[w[-1]].lnz[lnz_idx]
            new_lnz = np.asarray(new_lnz)

            for i in range(ntry):
                new_lnz[lnz_idx] += step * dlnz
                t = build_phases(ref=ref, lnz=new_lnz, **build_kws)
                if idx not in t.index:
                    right = t
                    break
            if right is None:
                raise RuntimeError('could not find right')

    return left, right


def _refine_bracket_spinodal_right(left, right,
                                   idx, idx_nebr=None,
                                   efac=1.0,
                                   nmax=30,
                                   vmax=1e5,
                                   vmin=0.0,
                                   ref=None,
                                   build_phases=None,
                                   build_kws=None,
                                   close_kws=None):
    """
    find refined bracket with efac<DeltabetaE_left<vmax and vmin<DeltabetaE_right<efac

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
            r = optimize.zeros.RootResults(
                root=None, iterations=i, function_calls=i, flag=1)
            return left, right, r

        ########
        #converged?
        if np.allclose(left.lnz, right.lnz, **close_kws):
            #we've reached a breaking point
            if doneLeft:
                #can't find a lower bound to efac, just return where we're at
                r = optimize.zeros.RootResults(
                    root=left.lnz, iterations=i + 1, function_calls=i, flag=0)
                for k, val in [('left', left), ('right', right),
                               ('doneleft', doneLeft), ('doneright','doneRight'),
                               ('info','all close and doneleft')]:
                    setattr(r, k, val)
                return left, right, r

            #elif not doneLeft and not doneRight:
            else:
                #all close, and no good on either end -> no spinodal
                r = optimize.zeros.RootResults(
                    root=None, iterations=i + 1, function_calls=i, flag=1)
                for k, val in [('left', left), ('right', right),
                               ('doneleft', doneLeft), ('doneright','doneRight'),
                               ('info','all close and doneleft')]:
                    setattr(r, k, val)
                for k, val in [('left', left), ('right', right),
                               ('info','all close and not doneleft')]:
                    setattr(r, k, val)
                return None, None, r

        # mid point phases
        lnz_mid = 0.5 * (left.lnz + right.lnz)
        mid    = build_phases(ref=ref, lnz =lnz_mid, **build_kws)
        # dw     = _get_dw(mid, idx, idx_nebr)
        dw     = mid.wlnPi.get_delta_w(idx, idx_nebr)

        if idx in mid.index and dw >= efac:
            left = mid
        else:
            right = mid

    # print('i', i)
    # print(ID)
    # print('left mu', left.mu)
    # print('right mu', right.mu)
    # print('dmu', left.mu - right.mu)
    # print('left DE', left.DeltabetaE_phaseIDs())
    # print('right DE', right.DeltabetaE_phaseIDs())
    # print('doneleft', doneLeft)
    # print('doneright', doneRight)
    # print('close', np.allclose(left.mu, right.mu, **close_kwargs))

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


def _solve_spinodal(lnz_in,
                    a, b,
                    idx, idx_nebr=None,
                    efac=1.0,
                    ref=None,
                    build_phases=None,
                    build_kws=None,
                    **kwargs):

    lnz_idx = lnz_in.index(None)
    if build_kws is None:
        build_kws = {}

    def f(x):
        lnz = lnz_in[:]
        lnz[lnz_idx] = x
        c = build_phases(ref=ref, lnz=lnz, **build_kws)

        dw = c.wlnPi.get_delta_w(idx, idx_nebr)

        out = dw - efac

        f._lnpi = c
        f._out = out

        return out

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)

    r.residual = f(xx)
    lnz = f._lnpi.lnz
    return lnz, r, f._lnpi


def _get_step(C, idx, idx_nebr):
    """
    find step value on

    if DeltabetaE[-1,ID] - DeltabetaE[0,ID]<0 -> step=+1 (go right)

    else step = -1
    """
    delta = (
        C[-1].wlnPi.get_delta_w(idx, idx_nebr) -
        C[0].wlnPi.get_delta_w(idx, idx_nebr)
    )

    if delta == 0:
        raise ValueError('could not determine step, delta==0')
    elif delta < 0.0:
        return +1
    else:
        return -1


def get_spinodal(C,
                 idx, idx_nebr=None,
                 efac=1.0,
                 dlnz=0.5,
                 vmin=0.0,
                 vmax=1e5,
                 ntry=20,
                 step=None,
                 nmax=20,
                 ref=None,
                 build_phases=None,
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
    assert(len(C) > 1)
    if build_phases is None:
        assert nphases_max is not None
        build_phases = get_default_PhaseCreator(nphases_max)
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

    lnz_idx = np.where(msk)[0][0]
    lnz_in = list(C[0].lnz[:])
    lnz_in[lnz_idx] = None

    #get initial bracket
    L, R = _initial_bracket_spinodal_right(CC,
                                           idx=idx,
                                           idx_nebr=idx_nebr,
                                           lnz_in=lnz_in,
                                           efac=efac,
                                           dlnz=dlnz,
                                           vmax=vmax,
                                           ntry=ntry,
                                           step=step,
                                           ref=ref,
                                           build_phases=build_phases,
                                           build_kws=build_kws)

    left, right, rr = _refine_bracket_spinodal_right(
        L, R, idx=idx, idx_nebr=idx_nebr,
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
        a, b = left.lnz[lnz_idx], right.lnz[lnz_idx]
        lnz, r, spin = _solve_spinodal(
            ref=ref, idx=idx, idx_nebr=idx_nebr,
            lnz_in=lnz_in,
            a=a, b=b,
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


from .core import CollectionPhases
from .cached_decorators import gcached

# NOTE : single create means this is only created once
@CollectionPhases.decorate_accessor('spinodals', single_create=True)
class Spinodals(object):
    def __init__(self, collection):
        self._c = collection

    @property
    def items(self):
        return self._items

    @gcached()
    def access(self):
        return CollectionPhases(self.items)

    def __getitem__(self, idx):
        return self._items[idx]


    def get(self, phase_ids, ref=None, build_phases=None, build_kws=None, nphases_max=None, inplace=True, append=True, force=False, **kwargs):

        if inplace and hasattr(self, '_items') and not force:
            raise ValueError('can reset inplace without force')
        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError('phase_ids must be an int or list')

        L = []
        info = []
        kwargs['full_output'] = True

        for idx in phase_ids:
            s, r = get_spinodal(ref=ref, C=self._c, idx=idx, build_phases=build_phases, build_kws=build_kws, nphases_max=nphases_max, **kwargs)
            L.append(s)
            info.append(r)

        L_masked = [x for x in L if x is not None]
        if append:
            self._c.extend(L_masked)
        if inplace:
            self._items = L
            self._info = info
        else:
            return L, info

    def index_collection(self, dtype=np.uint8):
        """
        return an array of same length as parent collection
        with each spinodal marked by a value > 0
        if this value == 0, then no spinod
        if this value >  0, then value - 1 is the spinodal index
        """

        out = []
        for rec, p in enumerate(self._c):
            val = 0
            for idx, s in enumerate(self.items):
                if s is p:
                    val = idx + 1
                    break
            out.append(val)
        return np.array(out, dtype=dtype)

    def set_by_index(self, index, nphase=None):
        """
        set by index array
        """
        features = np.unique(index[index > 0])
        if nphase is None:
            nphase = np.max(features)
        items = [None for i in range(nphase)]
        for feature in features:
            idx = np.where(index == feature)[0][0]
            items[feature - 1] = self._c[idx]
        self._items = items

    def assign_coords(self, da, name='spinodal', dtype=np.uint8):
        """
        add in index to dataarray
        """
        kws = {name : (self._c._CONCAT_DIM, self.index_collection(dtype=dtype))}
        return (
            da
            .assign_coords(**kws)
        )

    def from_dataarray(self, da, name='spinodal', nphase=None):
        """set from dataarray"""
        self.set_by_index(da[name].values, nphase=nphase)
 
