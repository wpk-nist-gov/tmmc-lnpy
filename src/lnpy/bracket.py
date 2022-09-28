"""
Set of routines to bracket solution of "thing"
"""


# def _bracket(
#     p_start,
#     build_phases,
#     check_condition,
#     sign=+1,
#     dlnz=0.5,
#     dfac=1.0,
#     ntry=20,
#     ref=None,
#     build_kws=None,
# ):

#     if build_kws is None:
#         build_kws = {}

#     new_lnz = p_start._get_lnz(build_phases.index)
#     dlnz_ = dlnz
#     for i in range(ntry):
#         new_lnz += sign * dlnz_
#         p = build_phases(new_lnz, ref=ref, **build_kws)
#         done, new_lnz, dlnz_ = check_condition(p, new_lnz, dlnz_)
#         if done:
#             return p, ntry

#         dlnz_ *= dfac

#     return None, ntry


# def _initial_bracket_spinodal_right(
#     C,
#     build_phases,
#     idx,
#     idx_nebr=None,
#     efac=1.0,
#     dlnz=0.5,
#     dfac=1.0,
#     vmax=1e5,
#     ntry=20,
#     step=+1,
#     ref=None,
#     build_kws=None,
# ):

#     if build_kws is None:
#         build_kws = {}
#     if efac <= 0:
#         raise ValueError("efac must be positive")

#     # Series representation of dw
#     s = C.wfe.get_dw(idx, idx_nebr)
#     if step < 0:
#         s = s.iloc[-1::-1]

#     # left bounds
#     s_idx = s[s > 0.0]
#     if len(s_idx) == 0:
#         raise ValueError("no phase {}".format(idx))
#     ss = s_idx[s_idx > efac]
#     if len(ss) > 0:
#         left = C.mloc[ss.index[[-1]]]
#     else:

#         def condition_left(p, new_lnz, dlnz_):
#             done = (
#                 idx in p._get_leve("phase")
#                 and p.wfe_signle.get_dw(idx, idx_nebr) > efac
#             )
#             return done, new_lnz, dlnz_

#         left, left_ntry = _bracket(
#             C.mloc[s_idx.index[[0]]],
#             build_phases,
#             condition_left,
#             sign=-1 * step,
#             dlnz=dlnz,
#             dfac=dfac,
#             ntry=ntry,
#             ref=ref,
#             build_kws=build_kws,
#         )
#     if left is None:
#         raise RuntimeError("could not find left")

#     # right
#     ss = s[s < efac]
#     if len(ss) > 0:
#         right = C.mloc[ss.index[[0]]]
#     else:

#         def condition_right(p, new_llnz, dlnz_):
#             done = (
#                 idx not in p._get_leve("phase")
#                 or t.wfe_phases.get_dw(idx, idx_nebr) < efac
#             )
#             return new, new_lnz, dlnz_

#         right, right_ntry = _bracket(
#             C.mloc[s.index[[-1]]],
#             build_phases,
#             condition_right,
#             sign=+1 * step,
#             dlnz=dlnz,
#             dfac=dfac,
#             ntry=ntry,
#             ref=ref,
#             build_kws=build_kws,
#         )
