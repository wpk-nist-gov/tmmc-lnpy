"""
routines to find binodal point between two phases
"""

import numpy as np
from scipy import optimize
from .segment import get_default_PhaseCreator


def get_binodal_point(IDs,
                      lnzA,
                      lnzB,
                      ref=None,
                      build_phases=None,
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
    lnzA,lnzB : arrays of shape (ncomp,)
        lnz arrays bracketing solution
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

    if build_phases is None:
        assert nphases_max is not None
        build_phases = get_default_PhaseCreator(nphases_max).build_phases
    if build_kws is None:
        build_kws = {}

    lnzA = np.asarray(lnzA)
    lnzB = np.asarray(lnzB)

    msk = lnzA != lnzB
    if msk.sum() != 1:
        raise ValueError('only one value can vary between lnzA and lnzB')

    lnz_idx = np.where(msk)[0][0]
    lnz_in = lnzA.copy()

    a, b = sorted([x[lnz_idx] for x in [lnzA, lnzB]])

    def f(x):
        lnz = lnz_in[:]
        lnz[lnz_idx] = x
        c = build_phases(ref=ref, lnz=lnz, **build_kws)
        f.lnpi = c
        print('lnz',lnz)
        # Omegas = c.omega_phase()
        # return Omegas[IDs[0]] - Omegas[IDs[1]]
        return c.xgce.omega().reindex(phase=IDs).diff('phase')

    xx, r = optimize.brentq(f, a, b, full_output=True, **kwargs)
    r.residual = f(xx)

    if full_output:
        return f.lnpi, r
    else:
        return f.lnpi




from .core import CollectionPhases
from .cached_decorators import gcached
import itertools

@CollectionPhases.decorate_accessor('binodals', single_create=True)
class Binodals(object):


    def __init__(self, collection):
        self._c = collection

    @property
    def items(self):
        return CollectionPhases(self._items)

    def __getitem__(self, idx):
        return self.items[idx]

    def get_pair(self, ids, lnzA=None, lnzB=None, spinodals=None, ref=None, build_phases=None, build_kws=None, nphases_max=None, **kwargs):

        if None in [lnzA, lnzB] and spinodals is None:
            spinodals = self._c.spinodals
        if lnzA is None:
            lnzA = spinodals[ids[0]].lnz
        if lnzB is None:
            lnzB = spinodals[ids[1]].lnz
        return get_binodal_point(ref=ref, IDs=ids, lnzA=lnzA, lnzB=lnzB, build_phases=build_phases, build_kws=build_kws, nphases_max=nphases_max,**kwargs)


    def get_all(self, phase_ids, spinodals=None, ref=None, build_phases=None, build_kws=None, nphases_max=None,  inplace=False, append=False, force=False, **kwargs):

        if inplace and not force and hasattr(self, '_items'):
            raise ValueError('can reset inplace without force')
        if isinstance(phase_ids, int):
            phase_ids = list(range(phase_ids))
        if not isinstance(phase_ids, list):
            raise ValueError('phase_ids must be an int or list')


        L = []
        info = []
        kwargs['full_output'] = True
        for ids in itertools.combinations(phase_ids, 2):
            s, r = self.get_pair(ref=ref, ids=ids, spinodals=spinodals,
                                 build_phases=build_phases,
                                 build_kws=build_kws,
                                 nphases_max=nphases_max, **kwargs)
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
        if this value == 0, then no binodal
        if this value >  0, then value - 1 is the binodal index
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

    def set_by_index(self, index):
        items = []

        features = np.unique(index[index > 0])
        for feature in features:
            idx = np.where(index == feature)[0][0]
            items.append(self._c[idx])
        self._items = items


    def assign_coords(self, da, name='binodal', dtype=np.uint8):
        """
        add in index to dataarray
        """
        kws = {name : (self._c._CONCAT_DIM, self.index_collection(dtype=dtype))}

        return (
            da
            .assign_coords(**kws)
        )

    def from_dataarray(self, da, name='binodal'):
        """set from dataarray"""
        self.set_by_index(da[name].values)









