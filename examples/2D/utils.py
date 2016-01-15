import h5py
import lnPi
import pandas as pd
import matplotlib.pyplot as plt

def get_spin_bin(ref,mu_in,x,spin_kwargs=None,bin_kwargs=None):
    """
    from reference lnpi_phase, create collection and calculate spinodals and binodals
    
    Parameters
    ----------
    ref : lnPi_phases

    mu_in : mu iterator generator

    x : variable mu range

    spin_kwargs,bin_kwargs : extra arguments to get_spinodals, get_binodals

    Returns
    -------
    C : lnPi_colllection with spinodals/binodals
    """

    if spin_kwargs is None: spin_kwargs={}
    if bin_kwargs is None: bin_kwargs = {}
    
    #initial estimate
    C = lnPi.lnPi_collection.from_mu(ref,mu_in,x)
    
    C.get_spinodals(**spin_kwargs)
    C.get_binodals(**bin_kwargs)
    
    return C






################################################################################
#read/write list of collections
################################################################################
def lnPi_collections_list_to_hdf(ref,lst,path_or_buff,key=None,overwrite=True):
    """
    write list of lnPi_collection(s) to h5py file
    """
    if isinstance(path_or_buff,(str,unicode)):
        p = h5py.File(path_or_buff)

    elif isinstance(path_or_buff,(h5py.File,h5py.Group)):
        p = path_or_buff
    else:
        raise ValueError('bad path_or_buff type %s'%(type(path_or_buff)))


    if key is None:
        group = p
    else:
        if key in p:
            if overwrite:
                del p[key]
            else:
                raise RuntimeError('key %s already exists'%key)
        group = p.create_group(key)
    

    if ref is not None:        
        ref.to_hdf(group,'lnpi_ref',overwrite=overwrite)
        
    collection_list = []
    for i,x in enumerate(lst):
        key = 'collection_%i'%i
        collection_list.append(key)
        x.to_hdf(group,key,ref=None,overwrite=overwrite)

    group.create_dataset('collection_list',data=np.array(collection_list))


def lnPi_collections_list_from_hdf(path_or_buff,key=None,ref=None,collection_key='collection_list',collection_list=None):
    """
    read list of lnPi_collection(s) from h5py file
    """

    if isinstance(path_or_buff,(str,unicode)):
        group = h5py.File(path_or_buff)

    elif isinstance(path_or_buff,(h5py.File,h5py.Group)):
        group = path_or_buff

    else:
        raise ValueError('bad path_or_buff')


    if key is not  None:
        group = group[key]


    if ref is None:
        ref = lnPi.lnPi_phases.from_hdf(group,'lnpi_ref')

    lst = []

    if collection_list is not None:
        itr = collection_list
    elif collection_key is not None:
        itr = group[collection_key]
    else:
        raise ValueError('need either collection_key or collection_list')
    
    for k in itr:
        lst.append(lnPi.lnPi_collection.from_hdf(group,k,ref=ref))


    return ref,lst


################################################################################
#DataFrames
################################################################################            
def get_spinodal_data(lst,merge=False,**kwargs):
    """
    get Dataframe of spinodal data
    """
    l = []
    for x in lst:
        for phaseID,s in enumerate(x.spinodals):
            if s is not None:
                if merge:
                    s = s.merge_phases(**kwargs)
                    
                d = dict(phaseID=phaseID)

                for comp,mu in enumerate(s.mu):
                    d['mu_%i'%comp] = mu

                for comp,mf in enumerate(s.molfracs_phaseIDs):
                    d['molfrac_%i'%comp] = s.molfracs_phaseIDs[phaseID,comp]
                    
                d['omega'] = s.Omegas_phaseIDs()[phaseID]
                l.append(d)

    return pd.DataFrame(l)


def get_binodal_data(lst,merge=False,**kwargs):
    """
    get DataFrame of binodal data
    """
    l = []
    for x in lst:
        for ID,b in enumerate(x.binodals):
            if b is not None:
                if merge:
                    b = b.merge_phases(**kwargs)

                d = dict(binodalID = ID)
                for comp,mu in enumerate(b.mu):
                    d['mu_%i'%comp] = mu

                for phaseID in range(b.base.num_phases_max):
                    dd = d.copy()
                    dd['phaseID'] = phaseID
                    for comp,mf in enumerate(b.molfracs_phaseIDs):
                        dd['molfrac_%i'%comp] = b.molfracs_phaseIDs[phaseID,comp]
                    dd['omega'] = b.Omegas_phaseIDs()[phaseID]
                    l.append(dd)

    return pd.DataFrame(l)



################################################################################
#plotting
################################################################################
def plot_omega_vs_molfrac(bino,spin,ls=['-','--'],colors=['k','k'],ax_labels=False,ax=None):
    """
    Note: linestyle -> [bin,spin], color -> phaseID
    """
    if ax is None:
        fig,ax = plt.subplots()
    
    if ax_labels:
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$-\Omega$')
        
    for phaseID,g in bino.groupby('phaseID'):
        ax.plot(g.molfrac_0,-g.omega,color=colors[phaseID],ls=ls[0],label=phaseID)
    
    for phaseID,g in spin.groupby('phaseID'):
        ax.plot(g.molfrac_0,-g.omega,color=colors[phaseID],ls=ls[1])
    return ax


def plot_mu0_vs_mu1(bino,spin,ls=['-','--'],colors=['b','r','g'],ax_labels=False,ax=None,
                    line_labels=['binodal','spin. 0','spin. 1']):
    """
    Note: linestyle -> [bin,spin], color -> bin, spin0,spin1
    """
    if ax is None:
        fig,ax = plt.subplots()
    
    if ax_labels:
        ax.set_xlabel(r'$\mu_0$')
        ax.set_ylabel(r'$\mu_1$')
    
    g = bino.query('phaseID==0')
    ax.plot(g.mu_0,g.mu_1,label=line_labels[0],ls=ls[0],color=colors[0])

    for i,(phaseID,g) in enumerate(spin.groupby('phaseID')):
        ax.plot(g.mu_0.values,g.mu_1.values,label=line_labels[1+i],ls=ls[1],color=colors[1+i])
    return ax
