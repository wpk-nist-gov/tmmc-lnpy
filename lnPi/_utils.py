"""
utility functions
"""

import numpy as np

def _pad_rows(Z,empty):
    #fill empty parts of Z
    ZR = Z.copy()
    for i in range(1,Z.shape[0]):
        msk = empty[i,:]
        last = Z[i,~msk][-1]
        ZR[i,msk] = last
    return ZR
    
def _pad_matrix(Z,empty):
    #fill empty parts of Z
    
    ZR = _pad_rows(Z,empty)
    ZC = _pad_rows(Z.T,empty.T).T
    ZRC = (ZR+ZC)*0.5
    return ZRC


def _interp_rows(Z,empty):
    ZR = Z.copy()
    x = np.arange(Z.shape[1])
    
    for i in range(0,Z.shape[0]):
        msk = empty[i,:]
        ZR[i,msk] = np.interp(x[msk],x[~msk],Z[i,~msk])
    return ZR

def _interp_matrix(Z,empty):
    ZR = _interp_rows(Z,empty)
    ZC = _interp_rows(Z.T,empty.T).T
    ZRC = 0.5*(ZR+ZC)
    return ZRC


# def _get_shift(shape,mu):
#     """
#     shift[i,j,...] = n1[i]*mu[1]+n2[j]*mu[2]+...
#     """
#     shift = np.zeros([],dtype=float)
#     for i,(s,m) in enumerate(zip(shape,mu)):
#         shift = np.add.outer(shift,np.arange(s)*m)
#     return shift

##################################################
#calculations
##################################################

def get_mu_iter(mu,x):
    """
    create a mu_iter object for varying a single mu

    Parameters
    ----------
    mu : list
        list with one element equal to None.  This is the component which will be varied
        For example, mu=[mu0,None,mu2] implies use values of mu0,mu2 for components 0 and 2, and 
        vary component 1

    x : array
        values to insert for variable component

    Returns
    -------
    ouptut : array of shape (len(x),len(mu))
       array with rows [mu0,mu1,mu2]
    """

    z = np.zeros_like(x)

    x = np.asarray(x)
    
    L = []
    for m in mu:
        if m is None:
            L.append(x)
        else:
            L.append(z+m)

    return np.array(L).T


##################################################
#utilities
##################################################

def sort_lnPis(input,comp=0):
    """
    sort list of lnPi  that component `comp` mol fraction increases

    Parameters
    ----------
    input : list of lnPi objects

    
    comp : int (Default 0)
     component to sort along

    Returns
    -------
    output : list of lnPi objects in sorted order
    """

    molfrac_comp = np.array([x.molfrac[comp] for x in input])

    order = np.argsort(molfrac_comp)

    output = [input[i] for i in order]

    return output
