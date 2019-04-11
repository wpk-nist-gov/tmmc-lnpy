import numpy as np


def build_phases(lnpi,num_phases_max=2,efac_cut=0.5,npmax_start=50,vmax=1e20,
                 DeltabetaE_kwargs={},
                 argmax_kwargs={},phases_kwargs={},ftag_phases=None,
                 force_nphases=True):
    """
    iteratively build phases
    """


    lnpi = lnpi.copy()
    lnpi.num_phases_max = npmax_start


    t = lnpi.to_phases(argmax_kwargs,phases_kwargs,ftag_phases)

    
    t.merge_phases(efac_cut,vmax=vmax,inplace=True,**DeltabetaE_kwargs)

    return t
    

    # while True:
        
    #     #check out DeltabetaE
    #     mat = t.DeltabetaE_matrix(vmax,**DeltabetaE_kwargs)
        

    #     #find min value
    #     min_val = np.nanmin(mat)

    #     if min_val>efac_cut:
    #         if not force_nphases:
    #             break
    #         elif t.nphase<=num_phases_max:
    #             break
            
        
    #     #else, we have some work to do.

    #     idx = np.where(mat==min_val)

    #     if len(idx[0])!=1:
    #         raise ValueError('more than one min found')


    #     idx_kill,idx_keep = [x[0] for x in idx]

    #     # print('mat')
    #     # print(mat)
    #     # print('min')
    #     # print(min_val)
    #     # print('idx')
    #     # print(idx)
    #     # print('idx')
    #     # print(idx_keep,idx_kill)
    #     # print('argmax')
    #     # print(t.argmax)
    #     # print('val')
    #     # print(t.base[t.argmax])
    #     # print('\n'*2)
        
    #     #get new phases
    #     new_mask = t[idx_keep].mask & t[idx_kill].mask
    #     tnew = t[idx_keep].new_mask(new_mask)

    #     #add in
    #     #t.phases = 'get'
    #     t._phases[idx_keep] = tnew
    #     t._phases.pop(idx_kill)

    #     #adjust argmax
    #     t.argmax = tuple(np.delete(x,idx_kill) for x in t.argmax)


    


    # if t.nphase > num_phases_max:
    #     print('bad nphases')

    # t.base.num_phases_max = num_phases_max

    # for p in t.phases:
    #     p.num_phases_max = num_phases_max


    # return t

        
        

    


    
    
