"""
utilities to work with lnPi(N)
"""

import numpy as np
from scipy.ndimage import filters

#import MaxSegmentation_igraph_or_nx as mseg
from collections import defaultdict
from collections import Iterable

from functools import wraps

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from skimage import draw
from skimage.graph import route_through_array
from scipy import ndimage as ndi


#decorator to kill cache
def cache_clear(f):
    def wrapper(*args):
        self = args[0]
        self._clear_cache()
        return f(*args)
    return wrapper


#decorator to cache property
def cache_prop(fn):
    key = fn.__name__
    @property
    def wrapper(self):
        if self._cached[key] is None:
            #print 'generating',key
            self._cached[key] = fn(self)
        return self._cached[key]
    return wrapper


#decorator to cache function
def cache_func(fn):
    key0 = fn.__name__
    @wraps(fn)
    def wrapper(*args):
        self = args[0]
        key = (key0,) + args[1:]
        if self._cached[key] is None:
            #print 'generating',key
            self._cached[key] = fn(*args)
        return self._cached[key]
    return wrapper


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




def _get_shift(shape,mu):
    """
    shift[i,j,...] = n1[i]*mu[1]+n2[j]*mu[2]+...
    """
    shift = np.zeros([],dtype=float)
    for i,(s,m) in enumerate(zip(shape,mu)):
        shift = np.add.outer(shift,np.arange(s)*m)
    return shift


# def _get_coords(shape,indexing='ij'):
#     args=tuple([np.arange(s) for s in shape])
#     coords = np.meshgrid(*args,indexing=indexing)
#     return coords


# _CACHE_ = {}

# def _get_cache_coords(shape):
#     if 'coords' not in _CACHE_ or shape!=_CACHE_['coords'].shape:
#         _CACHE_['coords'] = np.array(_get_coords(shape,indexing='ij'))
#     return _CACHE_['coords']
        

class lnPi(np.ma.MaskedArray):
    """
    class to store masked ln[Pi(n0,n1,...)].
    shape is (N0,N1,...) where Ni is the span of each dimension)

    Attributes
    ----------
    self : masked array containing lnPi

    mu : chemical potential for each component

    coords : coordinate array (ndim,N0,N1,...)

    pi : exp(lnPi)

    pi_norm : pi/pi.sum()

    Nave : average number of particles of each component

    molfrac : mol fraction of each component

    Omega : Omega system, relative to lnPi[0,0,...,0]


    set_data : set data array

    set_mask : set mask array


    argmax_local : local argmax (in np.where output form)

    get_phases : return phases


    ZeroMax : set lnPi = lnPi - lnPi.max()

    Pad : fill in masked points by interpolation

    adjust : ZeroMax and/or Pad

    
    reweight : create new lnPi at new mu

    new_mask : new object (default share data) with new mask

    add_mask : create object with mask = self.mask + mask

    smooth : create smoothed object
    

    from_file : create lnPi object from file
    
    from_data : create lnPi object from data array
    """

    def __new__(cls, data, mu=None,ZeroMax=False,Pad=False,num_phases_max=None,**kwargs):
        """
        constructor

        Parameters
        ----------
        data : array-like
         data for lnPi

        mu : array-like (Default None)
         if None, set mu=np.zeros(data.ndim)

        ZeroMax : bool (Default False)
         if True, shift lnPi = lnPi - lnPi.max()


        Pad : bool (Default False)
         if True, pad masked region by interpolation


        **kwargs : arguments to np.ma.array
         e.g., mask=...

        """
            
        obj = np.ma.array(data,**kwargs).view(cls)
#         if type(data) is np.ma.MaskedArray:
#             kwargs = dict({'mask':data.mask},**kwargs)
#             obj = np.ma.array(data.data,**kwargs).view(cls)
#         else:
#             obj = np.ma.array(data,**kwargs).view(cls)
            
        #copy fill value
        obj.set_fill_value(getattr(data,'fill_value',None))

        #overide?
        fv = kwargs.get('fill_value',None)
        if fv is not None:
            obj.set_fill_value(fv)

        obj.mu = mu
        obj.num_phases_max = num_phases_max


        obj.adjust(ZeroMax=ZeroMax,Pad=Pad,inplace=True)

        return obj
    

    ##################################################
    #caching
    def __array_finalize__(self,obj):
        super(lnPi,self).__array_finalize__(obj)
        self._clear_cache()
        

    def _clear_cache(self):
        self._cached = defaultdict(lambda : None)        
        


    ##################################################
    #properties
    @property
    def mu(self):
        mu = self._optinfo.get('mu',None)
        if mu is None:
            mu = np.zeros(self.ndim)
        mu = np.atleast_1d(mu)
        
        if len(mu) != self.ndim:
            raise ValueError('bad len on mu %s'%mu)
        return mu

    @mu.setter
    def mu(self,val):
        if val is not None:
            self._optinfo['mu'] = val


    @property
    def num_phases_max(self):
        return self._optinfo.get('num_phases_max',None)

    @num_phases_max.setter
    def num_phases_max(self,val):
        if val is not None:
            self._optinfo['num_phases_max'] = val
        

    @property
    def coords(self):
        return np.indices(self.shape)


    @cache_prop
    def pi(self):
        """
        basic pi = exp(lnpi)
        """
        pi = np.exp(self)
        return pi

    @property
    def pi_norm(self):
        p = self.pi
        return p/p.sum()
    

    @cache_prop
    def Nave(self):
        #<N_i>=sum(N_i*exp(lnPi))/sum(exp(lnPi))
        N = (self.pi_norm*self.coords).reshape(self.ndim,-1).sum(axis=-1).data
        return N

    @property
    def molfrac(self):
        n = self.Nave
        return n/n.sum()


    def Omega(self,zval=None):
        """
        get omega = zval - ln(sum(pi))

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """

        if zval is None:
            zval = self.data.ravel()[0]

        omega = zval - np.log(self.pi.sum())

        return omega

    @property
    def ncomp(self):
        return self.ndim
    

    ##################################################
    #setters
    @cache_clear
    def set_data(self,val):
        self.data[:] = val


    @cache_clear
    def set_mask(self,val):
        self.mask[:] = val



    ##################################################
    #maxima
    def _peak_local_max(self,min_distance=1,**kwargs):
        """
        find local maxima using skimage.feature.peak_local_max

        Parameters
        ----------
        min_distance : int (Default 1)
          min_distance parameter to peak_local_max

        **kwargs : arguments to peak_local_max (see docs)
         Defaults to min_distance=1,exclude_border=False
        
        Returns
        -------
        out : tupe of ndarrays
         indices of self where local max

        n : int
         number of maxima found
        """

        kwargs = dict(dict(exclude_border=False,labels=~self.mask),**kwargs)

        data = self.data - np.nanmin(self.data)

        x = peak_local_max(data,min_distance=min_distance,**kwargs)

        if kwargs.get('indices',True):
            #return indices
            x = tuple(x.T)
            n = len(x[0])
        else:
            n = x.sum()

        return x,n


    def argmax_local(self,min_distance=[10,15,20,25,30],smooth='fail',smooth_kwargs={},info=False,**kwargs):
        """
        find local max with fall backs min_distance and filter

        Parameters
        ----------
        min_distance : int or iterable (Default 15)
            min_distance parameter to self.peak_local_max. 
            if min_distance is iterable, if num_phase>num_phase_max, try next
       

        smooth : bool or str (Default 'fail')
            if True, use smooth only
            if False, no smooth
            if 'fail', do smoothing only if non-smooth fails

        smooth_kwargs : dict
            extra arguments to self.smooth

        info : bool (Default False)
            if True, return out_info

        **kwargs : extra arguments to peak_local_max

        Returns
        -------
        out : tuple of ndarrays
            indices of self where local max

        out_info : tuple (min_distance,smooth)
            min_distance used and bool indicating if smoothing was used


        """
        num_phases_max = self.num_phases_max


        if not isinstance(min_distance,Iterable):
            min_distance = [min_distance]

        if smooth is True:
            filtA = [True]
        elif smooth is False:
            filtA = [False]
        elif smooth.lower()=='fail':
            filtA = [False,True]
        else:
            raise ValueError('bad parameter %s'%(smooth))

        for filt in filtA:
            if filt:
                xx = self.Pad().smooth(**smooth_kwargs)
            else:
                xx = self

            for md in min_distance:
                x,n = xx._peak_local_max(min_distance=md,**kwargs)

                if n <= num_phases_max:
                    if info:
                        return x,(md,filt)
                    else:
                        return x

        #if got here than error
        raise RuntimeError('%i maxima found greater than %i'%(n,num_phases_max))    


    
    

    ##################################################
    #segmentation
    def _get_labels(self,indices,structure='set',size=3,footprint=None,**kwargs):
        """
        get labels from watershed segmentation

        Parameters
        ----------
        indices : tuple of ndarrays
         indices of location of points to segment to (e.g., maxima).
         len==ndim, seg_indices[i].shape == (nsegs)
         e.g., from output of self.peak_local_max

        structure : array or str or None
         structure passed to ndimage.label.
         if None, use default.
         if 'set', use np.one((3,)*data.ndim)
         else, use structure
        """

        markers,n = _indices_to_markers(indices,self.shape,structure)
        labels = _labels_watershed(-self.data,markers,mask=(~self.mask),size=size,footprint=footprint,**kwargs)

        return labels


    def _get_list_regmask(self,regmask,SegLenOne=False):
        """
        create list of lnPi objects with mask = self.mask + regmask[i]
        """
        if not SegLenOne and len(regmask)==1:
            return None#[self]
        else:
            return  [self.add_mask(r) for r in regmask]
        

    def _get_list_labels(self,labels,SegLenOne=False,**kwargs):
        """
        create list of lnpi's from labels
        """
        regmask = labels_to_masks(labels,**kwargs)
        return self._get_list_regmask(regmask,SegLenOne)
    

    def _get_list_indices(self,indices,SegLenOne=False,labels_kwargs={},masks_kwargs={}):
        """
        create list of lnpi's from indices
        """
        if not SegLenOne and len(indices[0])==1:
            return None#[self]
        else:
            labels = self._get_labels(indices,**labels_kwargs)
            return self._get_list_labels(labels,SegLenOne,**masks_kwargs)


    def get_phases(self,indices=None,regmask=None,labels=None,
                   get=False,
                   SegLenOne=False,
                   sort=True,comp=0,
                   phases=True,
                   argmax_kwargs={},
                   labels_kwargs={},
                   masks_kwargs={},**kwargs):
        """
        get list of lnPi objects by regmask,labels, or indices

        Parameters
        ----------
        indices : tuple of arrays
            indices of self where features are located
            (labels = self._get_labels(**label_kwargs))

        regmask : list of masks

        labels : array of labels
            create an lnPi object for each label.
            regmask = labels_to_masks(labels,**masks_kwargs)

        get : bool (Default False)
            if True, indices = self.argmax_local(**argmax_kwargs)
            i.e., do total get in one step

        SegLenOne : bool (Default False)
          if False and nphase==1, phases=[self]

        sort : bool (Default True)
         if True, sort output
        
        comp : int (Default 0)
         component to sort along

        phases: bool (Default True)
         if True, return lnPi_phases object
         otherwise, return list of lnpi objects
        
        argmax_kwargs : dict of options to self.argmax_local

        labels_kwargs : dict of options to self._get_labels

        masks_kwargs : dict of optoins to labels_to_masks

        **kwargs : extra kwargs to creation of new lnpi objects

        Returns
        -------
        output : lnPi_phases or list or None
         if phases is True, return lnPi_phases object.
         otherwise, return list of lnPi objects. if len(regmask)==1,
         return None
         
        Notes
        -----
        One of regmask,labels, or indices must be specified
        """
        if get:
            indices = self.argmax_local(**argmax_kwargs)
        
        
        if regmask is not None:
            output = self._get_list_regmask(regmask,SegLenOne)
        elif labels is not None:
            output = self._get_list_labels(labels,SegLenOne,**mask_kwargs)
        elif indices is not None:
            output = self._get_list_indices(indices,SegLenOne,labels_kwargs,masks_kwargs)
        else:
            raise ValueError('must specify one of regmask, labels, indices')


        if output is not None and sort:
            output = sort_lnPis(output,comp)    


        if phases:
            return lnPi_phases(base=self,phases=output,argmax=indices)
        else:
            return output



        
    ##################################################
    #adjusters
    def ZeroMax(self,inplace=False):
        """
        shift so that lnpi.max() == 0
        """

        if inplace:
            y = self
        else:
            y = self.copy()
            
        
        shift = self.max()

        y.set_data(y.data-shift)

        
        if not inplace:
            return y
        
    
    def Pad(self,inplace=False):
        """
        pad self.data rows and columns by last value
        """
        data = _interp_matrix(self.data,self.mask)
        
            
        if inplace:
            y = self
        else:
            y = self.copy()

        y.set_data(data)
        
        if not inplace:
            return y



    def adjust(self,ZeroMax=False,Pad=False,inplace=False):
        """
        do multiple adjustments in one go
        """

        if inplace:
            Z = self
        else:
            Z = self.copy()

        if ZeroMax:
            Z.ZeroMax(inplace=True)

        if Pad:
            Z.Pad(inplace=True)
                   

        if not inplace:
           return Z
                   
                


    ##################################################
    #new object/modification
    def reweight(self,mu,beta=1.0,ZeroMax=False,Pad=False):
        """
        get lnpi at new mu
        
        Parameters
        ----------
        mu : array-like 
            chem. pot. for new state point
            
        beta : float
            inverse temperature

        ZeroMax : bool

        Pad : bool
            
        Returns
        -------
        lnPi(mu)
        """
        
        Z = self.copy()
        Z.mu = mu
        
        dmu = Z.mu - self.mu
        
        s = _get_shift(self.shape,dmu)
        Z.set_data(Z.data+s)


        Z.adjust(ZeroMax=ZeroMax,Pad=Pad,inplace=True)
                
        return Z


    
    def new_mask(self,mask=None,**kwargs):
        return lnPi(self.data,mask=mask,mu=self.mu,num_phases_max=self.num_phases_max,**kwargs)

    
    def add_mask(self,mask,**kwargs):
        """
        logical or of self.mask and mask

        Note, if want a copy, pass copy=True
        """
        return self.new_mask(mask=mask+self.mask,**kwargs)


    def smooth(self,sigma=4,mode='nearest',truncate=4,inplace=False,ZeroMax=False,Pad=False,**kwargs):
        """
        apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool (Default False)
         if True, do inplace modification.


        **kwargs : (Default sigma=4, mode='nearest',truncate=4)
         arguments to filters.gaussian_filter
        """

        kwargs = dict(dict(sigma=4,mode='nearest',truncate=4),**kwargs)
        
        if inplace:
            Z = self
        else:
            Z = self.copy()

        Z._clear_cache()

        Z.adjust(ZeroMax=ZeroMax,Pad=Pad,inplace=True)

        filters.gaussian_filter(Z.data,output=Z.data,**kwargs)

        if not inplace:
            return Z




        
    ##################################################
    #create from file/etc
    @staticmethod
    def from_file(filename,mu=None,loadtxt_kwargs={},**kwargs):
        """
        load filename into lnPi object
        
        Parameters
        ----------
        filename : str
            name of file to read
            
        mu : array-like or None
            chemical potential.  If None, default to Zero for each
            component
            
        load_txt_kwargs : dict
            keyword arguments for numpy.loadtxt
            
        **kwargs : kwargs to lnPi constructor
        
        Returns
        -------
        lnPi object
        """
        
        data = np.loadtxt(filename,**loadtxt_kwargs)
        
        return lnPi.from_data(data,mu,**kwargs)
        
        
    
    @staticmethod
    def from_data(data,mu=None,num_phases_max=None,**kwargs):
        """
        parse data into lnpi masked array
        
        Parameters
        ----------
        data : array of form (n0,n1,...,nd,lnPi)
        
        mu : array-like (d,)
            chem. pot. for each component

        num_phases_max : int or None (default None)
            max number of phases
            
        **kwargs : arguments to lnPi constructor
        """
        
        ndim = data.shape[-1]
        dataT = data.T
    
        z = dataT[-1,:]

        xx = dataT[:-1,:].astype(int)
        shape = tuple(xx.max(axis=-1)+1)
        
        xx = tuple(xx)
    
    
        Z = np.zeros(shape,dtype=data.dtype)
        Z[xx] = z

        empty = np.ones(shape,dtype=bool)
        empty[xx] = False
        
        return lnPi(Z,mask=empty,mu=mu,num_phases_max=num_phases_max,**kwargs)



def tag_phases_binary(x):
    if x.base.num_phases_max !=2:
        raise ValueError('bad tag function')

    L = []
    for p in x.phases:
        if p.molfrac[0]<0.5:
            val = 0
        else:
            val = 1
        L.append(val)

    return np.array(L)
        

    
    


class lnPi_phases(object):
    """
    object containing lnpi base and phases
    """


    def __init__(self,base,phases='get',argmax='get',
                 argmax_kwargs={},
                 phases_kwargs={},
                 ftag_phases=tag_phases_binary):
        """
        object to store base and phases

        Parameters
        ----------
        base : lnPi object
            base object.  Not parsed to phases

        phases : list of lnPi objects, None, or str
            if list of lnPi objects, each corresponds to an independent phase.
            if None, implies single phase (phases=[base])
            if str and 'get', get phases on demand

        argmax : tuple of arrays (indicies) or str
            indices of local max locations.
            if str and 'get', get argmax on demand
        
        argmax_kwargs : dict 
            if argmax=='get', self.argmax = self.base.argmax_local(**argmax_kwargs)

        phases_kwargs : dict
            if phases=='get', self.phases = self.base._get_list_indices(self.argmax,**phases_kwargs)

        ftag_phases : function (default tag_phases_binary)
            function which returns integer phase_id for each phase
            ...
            ftag_phases(self):
                returns [phase_id(i) for i in len(phases)]
        """
        
        self.base = base
        self.phases = phases
        self.argmax = argmax

        self._ftag_phases = ftag_phases

        

        self._argmax_kwargs = argmax_kwargs
        self._phases_kwargs = phases_kwargs

    ##################################################
    #properties
    @property
    def base(self):
        return self._base

    @base.setter
    def base(self,val):
        if type(val) is not lnPi:
            raise ValueError('base must be type lnPi %s'%(type(val)))
        self._base = val


    @property
    def argmax(self):
        #set on access
        if self._argmax == 'get':
            self.argmax = self.base.argmax_local(**self._argmax_kwargs)
        return self._argmax

    @argmax.setter
    def argmax(self,val):
        if val=='get':
            pass
        else:
            assert self.base.ndim == len(val)
            assert len(val[0]) <= self.base.num_phases_max
            
        self._argmax = val
        

    @property
    def phases(self):
        if self._phases == 'get':
            self.phases = self._base._get_list_indices(self.argmax,**self._phases_kwargs)

        if self._phases is None:
            return [self.base]
        else:
            return self._phases

    @phases.setter
    def phases(self,phases):
        if phases is None or phases == 'get':
            self._phases = phases
        else:
            #check
            if not isinstance(phases,(tuple,list)):
                raise ValueError('phases must be a list or tuple')

            for i,p in enumerate(phases):
                if type(p) is not lnPi:
                    raise ValueError('element %i of phases must be lnPi'%(i))

                if np.any(self.base.mu!=p.mu):
                    raise ValueError('bad mu between base and comp %i: %s, %s'%(i,base.mu,p.mu))
            self._phases = phases


    @property
    def phaseIDs(self):
        return self._ftag_phases(self)


    def phaseIDs_to_indicies(self,IDs):
        """
        convert IDs to indicies in self.phases
        """
        l = list(self.phaseIDs)
        return np.array([l.index(i) for i in IDs])
 

    def __len__(self):
        return len(self.argmax[0])

    def __getitem__(self,i):
        return self.phases[i]

    
    @property
    def nphase(self):
        return len(self)


    @property
    def masks(self):
        return np.array([p.mask for p in self.phases])

    @property
    def labels(self):
        return masks_to_labels(self.masks,feature_value=False)


    @property
    def molfracs(self):
        return np.array([x.molfrac for x in self])


    @property
    def molfracs_phaseIDs(self):
        molfrac = np.zeros((self.base.num_phases_max,self.base.ndim),dtype=float)*np.nan

        for i,mf in zip(self.phaseIDs,self.molfracs):
            molfrac[i,:] = mf
        return molfrac
         

    def Omegas(self,zval=None):
        return np.array([x.Omega(zval)  for x in self])


    def Omegas_phaseIDs(self,zval=None):
        Omegas = np.zeros((self.base.num_phases_max,),dtype=float)*np.nan

        for i,x in zip(self.phaseIDs,self.Omegas(zval)):
            Omegas[i] = x
        return Omegas


        

    @property
    def pis(self):
        return np.array([x.pi for x in self])

    @property
    def pi_norms(self):
        return np.array([x.pi_norm for x in self])

    @property
    def mu(self):
        return self.base.mu



    




    # ##################################################
    # #fill
    # def set_argmax(self,value='get',**kwargs):
    #     """
    #     set argmax by value
    #     """
    #     if value=='get':
    #         value = self.base.argmax_local(**kwargs)

        
    #     self.argmax = value
    #     self._check_other()
        

    # def set_phases(self,value='get',**kwargs):
    #     """
    #     set phases by value
    #     """
    #     if value=='get':
    #         value = self.base._get_list_indices(self.argmax,**kwargs)

    #     self.phases = value
    #     self._check_phases()


    # def set_all(self,argmax='get',phases='get',argmax_kwargs={},phases_kwargs={}):
    #     self.fill_argmax(indices,**argmax_kwargs)
    #     self.fill_phases(indices,**phases_kwargs)
    


    ##################################################
    #query
    def _get_boundaries(self,pair,mode='thick',connectivity=None,**kwargs):
        """
        get the boundary between phase pair

        Parameters
        ----------
        pair : tuple (i0,i1) of phase indices to locate boundaries

        mode : string (Default 'thick')
         mode passed to find_boundaries

        connectivity : int (Default None)
         if None, use self.ndim 

        **kwargs : extra arguments to find_boundaries

        Returns
        -------
        output : array of shape self.base.shape of bools
          output==True at boundary locations
        """

        if connectivity is None:
            connectivity = self.base.ndim
            
        

        b = []

        for i in pair:
            p = self.phases[i]
            msk = ~p.mask
            b.append(find_boundaries(msk.astype(int),mode=mode,connectivity=connectivity,**kwargs))

        b = np.prod(b,axis=0).astype(bool)
        b *= ~self.base.mask

        return b


    def betaEmin(self,pair=(0,1)):
        """
        value of beta E at minima
        
        beta E_min ~ -max{lnPi}
        """

        return -self.base[self.argmax].filled()[np.asarray(pair)]


    def betaEtransition(self,pair=(0,1),**kwargs):
        """
        Transition point energy.  

        minimum value of energy at boundary between phases
        beta E_boundary = - max{lnPi}_{along boundary}
        """

        b = self._get_boundaries(pair,**kwargs)


        return -(self.base[b].max())


    def DeltabetaE(self,pair=(0,1),**kwargs):
        """
        betaE_transition - betaEmin
        """
        
        return self.betaEtransition(pair,**kwargs) - self.betaEmin(pair)


    def DeltabetaE_phaseIDs(self,IDs=(0,1),**kwargs):
        """
        find delta E between basin and transition points

        IDs = pairID of each phaseIDs
        
        return in order of IDs
        """

        pair = self.phaseIDs_to_indicies(IDs)

        return self.DeltabetaE(pair,**kwargs)
        



    def _betaEtransition_line(self,pair=(0,1),connectivity=None):
        """
        find transition energy from line connecting location of maxima
        """

        args = tuple(np.array(self.argmax).T.flatten())

        img = np.zeros(self.base.shape,dtype=int)
        img[draw.line(*args)] = 1

        if connectivity is None:
            msk = img.astype(bool)
        else:
            msk = find_boundaries(img,connectivity=connectivity)

        return -(self.base[msk].min())


    def _betaEtransition_path(self,pair=(0,1),**kwargs):
        """
        construct a path connecting maxima
        """

        idx = np.array(self.argmax).T
        
        d = -self.base.data.copy()
        d[self.base.mask] = d.max()

        i,w = route_through_array(d,idx[pair[0]],idx[pair[1]],**kwargs)

        i = tuple(np.array(i).T)

        return -(self.base[i].min())


    ##################################################
    #repr
    def _repr_html_(self):
        if self._phases=='get':
            x = self._phases
        else:
            x = self.nphase
        return 'lnPi_phases: nphase=%s, mu=%s'%(x,self.base.mu)
        
        


################################################################################
#collection
################################################################################
class lnPi_collection(object):
    """
    class containing several lnPis
    """

    def __init__(self,lnpis,argmax_kwargs={},phases_kwargs={},ftag_phases=tag_phases_binary):
        """
        Parameters
        ----------
        lnpis : iterable of lnpi objects
        """

        self._argmax_kwargs = argmax_kwargs
        self._phases_kwargs = phases_kwargs
        self._ftag_phases = ftag_phases
        

        self.lnpis = lnpis


    ##################################################
    #setters
    def _parse_lnpi(self,x):
        if type(x) is lnPi:
            #create placeholder from x
            return  lnPi_phases(base=x,
                             argmax_kwargs=self._argmax_kwargs,
                             phases_kwargs=self._phases_kwargs,
                             ftag_phases=self._ftag_phases)
        elif type(x) is lnPi_phases:
            return x
        else:
            raise ValueError('bad value while parsing element %i'%(i))
            
    def _parse_lnpis(self,lnpis):
        """
        from a list of lnpis, return list of lnpi_phases
        """

        if not isinstance(lnpis,(list,tuple)):
            raise ValueError('lnpis must be list or tuple')
        
        L = []
        for i,x in enumerate(lnpis):
            L.append(self._parse_lnpi(x))
            
        return L
            
    ##################################################
    #properties

    def __len__(self):
        return len(self.mus)

    @property
    def shape(self):
        return (len(self),)

    @property
    def lnpis(self):
        return self._lnpis

    @lnpis.setter
    def lnpis(self,val):
        self._lnpis = self._parse_lnpis(val)


    def append(self,val):
        self._lnpis.append(self._parse_lnpi(val))


    def extend(self,x):
        if isinstance(x,list):
            self._lnpis.extend(self._parse_lnpis(x))
        elif isinstance(x,lnPi_collection):
            self._lnpis.extend(x._lnpis)
        else:
            raise ValueError('only lists or lnPi_collections can be added')


    def extend_by_mu(self,ref,mu,x,**kwargs):
        """
        extend self my mu values
        """

        new = lnPi_collection.from_mu(ref,mu,x,
                                      self._argmax_kwargs,
                                      self._phases_kwargs,
                                      self._ftag_phases,**kwargs)
        self.extend(new)
        

    def __add__(self,x):
        if isinstance(x,list):
            L = self._lnpis + self._parse_lnpis(x)
        elif isinstance(x,lnPi_collection):
            L = self._lnpis + x._lnpis
        else:
            raise ValueError('only lists or lnPi_collections can be added')
            
        return lnPi_collection(L,self._argmax_kwargs,self._phases_kwargs,self._ftag_phases)


    def sort_by_mu(self,comp=0,inplace=False):
        """
        sort self.lnpis by mu[:,comp]
        """

        order = np.argsort(self.mus[:,comp])
        L = []
        for i in order:
            L.append(self._lnpis[i])

        if inplace:
            self._lnpis = L
        else:
            return lnPi_collection(L,self._argmax_kwargs,self._phases_kwargs,self._ftag_phases)
    
        
    
    def __getitem__(self,i):
        if type(i) is int:
            return self.lnpis[i]
        
        elif type(i) is slice:
            L = self.lnpis[i]
            
        elif isinstance(i,(list,np.ndarray)):
            idx = np.array(i)

            if np.issubdtype(idx.dtype,np.integer):
                L = [self.lnpis[j] for j in idx]

            elif np.issubdtype(idx.dtype,np.bool):
                assert idx.shape == self.shape
                L = [xx for xx,mm in zip(self.lnpis,idx) if mm]
            
            else:
                raise KeyError('bad key')
                

        return lnPi_collection(L,
                               self._argmax_kwargs,
                               self._phases_kwargs)



    


    ##################################################
    #calculations/props
    @property
    def mus(self):
        return np.array([x.mu for x in self.lnpis])

    @property
    def nphases(self):
        return  np.array([x.nphase for x in self.lnpis])


    @property
    def molfracs(self):
        return [x.molfracs for x in self.lnpis]

    @property
    def molfracs_phaseIDs(self):
        return np.array([x.molfracs_phaseIDs for x in self.lnpis])
    

    def Omegas_phaseIDs(self,zval=None):
        return np.array([x.Omegas_phaseIDs(zval) for x in self.lnpis])



    def DeltabetaE(self,pair=(0,1),**kwargs):
        return np.array([x.DeltabetaE(pair,**kwargs) for x in self])

    def DeltabetaE_phaseIDs(self,IDs=(0,1),**kwargs):
        return np.array([x.DeltabetaE_phaseIDs(IDs,**kwargs) for x in self])
    
    

    def get_binodal_interp(self,mu_axis,phases=(0,1)):
        """
        get position of Omega[i]==Omega[j] by interpolation
        """
        
        idx = np.asarray(phases)

        assert(len(idx)==2)
        
        x = self.Omegas_padded()[:,idx]
        msk = np.prod(~np.isnan(x),axis=1).astype(bool)
        assert(msk.sum()>0)


        diff = x[msk,0] - x[msk,1]

        mu = self.mus[msk,mu_axis]
        i = np.argsort(diff)
        
        
        return np.interp(0.0,diff[i],mu[i])


    def get_binodal_solve(self,mu_axis,phases=(0,1)):
        pass
    
    
    ##################################################
    def _repr_html_(self):
        return 'lnPi_collection: %s'%len(self)



    ##################################################
    #builders
    ##################################################
    @staticmethod
    def from_mu_iter(ref,mus,
                     argmax_kwargs={},phases_kwargs={},
                     ftag_phases=tag_phases_binary,**kwargs):
        """
        build lnPi_collection from mus

        Parameters
        ----------
        ref : lnpi object
            lnpi to reweight to get list of lnpi's

        mus : iterable
            chem. pots. to get lnpi


        argmax_kwargs : dict 
            argmax_kwargs parameter in lnPi_collection()

        phases_kwargs : dict
            phases_kwargs parameter in lnPi_collection
        
        ftag_phases : function
            tag function for phase_ids

        **kwargs : arguments to ref.reweight
        
        Returns
        -------
        out : lnPi_collection object
        """

        kwargs = dict(dict(ZeroMax=True),**kwargs)
        L = []
        for mu in mus:
            L.append( lnPi_phases(ref.reweight(mu,**kwargs)))


        return lnPi_collection(L,argmax_kwargs=argmax_kwargs,phases_kwargs=phases_kwargs,ftag_phases=tag_phases_binary)
                      

    @staticmethod
    def from_mu(ref,mu,x,
                argmax_kwargs={},phases_kwargs={},
                ftag_phases=tag_phases_binary,**kwargs):
        """
        build lnPi_collection from mu builder

        Parameters
        ----------
        ref : lnpi object
            lnpi to reweight to get list of lnpi's

        mu : list
            list with one element equal to None.  
            This is the component which will be varied
            For example, mu=[mu0,None,mu2] implies use values 
            of mu0,mu2 for components 0 and 2, and vary component 1

        x : array
            values to insert for variable component

        argmax_kwargs : dict 
            argmax_kwargs parameter in lnPi_collection()

        phases_kwargs : dict
            phases_kwargs parameter in lnPi_collection
        
        ftag_phases : function
            tag function for phase_ids

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : lnPi_collection object
        """
       
        mus = get_mu_iter(mu,x)
        return lnPi_collection.from_mu_iter(ref,mus,
                                            argmax_kwargs,phases_kwargs,
                                            ftag_phases,**kwargs)
    
        
                      

    


##################################################
#calculations
##################################################
from scipy import optimize

def get_binodal_point(reference,mu_in,a,b,
                      phases=(0,1),
                      reweight_kwargs={},
                      phases_kwargs={},
                      **kwargs):
    """
    calculate binodal point where Omega0==Omega1
    """
    
    idx = mu_in.index(None)
    reweight_kwargs = dict(dict(ZeroMax=True),**reweight_kwargs)

    phases_kwargs = dict(dict(get=True),**phases_kwargs)
    
    def f(x):
        mu = mu_in[:]
        mu[idx] = x
        c = reference.reweight(mu,**reweight_kwargs).get_phases(**phases_kwargs)
        return c[phases[0]].Omega() - c[phases[1]].Omega()
    
    xx = optimize.brentq(f,a,b,**kwargs)
    
    mu = mu_in[:]
    mu[idx] = xx
    res = f(xx)
    return mu,res


def get_spinodal_point(ref,mu_in,a,b,
                       phases=(0,1)
            


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





##################################################
#segmentation functions
##################################################
def _indices_to_markers(indices,shape,structure='set',**kwargs):
    """
    create markers array from feature indices

    Parameters
    ----------
    indices : tuple of arrays
        indices of features

    shape : tuple
        shape of resulting markers array

    structure : array or str or None
        structure passed to ndimage.label.
        if None, use default.
        if 'set', use np.one((3,)*data.ndim)
        else, use structure
    
    Returns
    -------
    markers : array of shape self.shape
        roughly, np.where(makers==i) = seg_indices[0][i],seg_indices[1][i],....
    
    num_feature : int
        Number of objects found
    """

    input = np.zeros(shape,dtype=bool)
    input[indices] = True

    return _mask_to_markers(input,structure,**kwargs)


def _mask_to_markers(input,structure='set',**kwargs):
    """"
    create markers array from input

    Parameters
    ----------
    input : array_like
        An array-like object to be labeled.  Any non-zero values in `input` are
        counted as features and zero values are considered the background.

    structure : array or str or None
        structure passed to ndimage.label.
        if None, use default.
        if 'set', use np.one((3,)*data.ndim)
        else, use structure
    
    Returns
    -------
    markers : array of shape self.shape
        roughly, np.where(makers==i) = seg_indices[0][i],seg_indices[1][i],....
    
    num_feature : int
        Number of objects found
    """

    #make labels
    if isinstance(structure,(str,unicode)) and structure=='set':
        structure = np.ones((3,)*input.ndim)

    return ndi.label(input,structure=structure,**kwargs)
    



def _labels_watershed(data,markers,mask,size=3,footprint=None,**kwargs):
    """"
    perform watershed segmentation on data

    Parameters
    ----------
    data : array
     data to segment

    markers: ndarray of the same shape as `data`
        An array marking the basins with the values to be assigned in the
        label matrix. Zero means not a marker. This array should be of an
        integer type.
     
    mask: ndarray of bools or 0s and 1s, optional
     Array of same shape as `image`. Only points at which mask == True
     will be labeled.

    size : int (Default 3)
     set footprint=np.ones((size,)*data.ndim,dtype=bool)

    footprint : array
     connectivity footprint array

    **kwargs : keyword arguments to watershed

    """

    

    #get labels from watershed
    connectivity=None
    if size is not None:
        connectivity = np.ones((size,)*data.ndim)
        
    if footprint is not None:
        connectivity = footprint

    labels = watershed(data,markers,connectivity=connectivity,mask=mask,**kwargs)

    return labels




##################################################
#labels/masks utilities
##################################################
def labels_to_masks(labels,num_feature=None,include_boundary=False,feature_value=False,**kwargs):
    """
    convert labels array to list of masks

    Parameters
    ----------
    labels : array of labels to analyze

    num_features : number of features to analyze (Default None)
        if None, get num_feature from labels

    include_boundary : bool (Default False)
        if True, include boundary regions in output mask


    feature value : bool (Default False)
        value which indicates feature.
        False is MaskedArray convension
        True is image convension

    **kwargs : arguments to find_boundary if include_boundary is True
    mode='outer', connectivity=labels.ndim

    Returns
    -------
    output : list of masks of same shape as labels
        mask for each feature

    """


    if include_boundary:
        kwargs = dict(dict(mode='outer',connectivity=labels.ndim),**kwargs)
    
    if num_feature is None:
        num_feature = labels.max()


    output = []

    for i in range(1,num_feature+1):
        m = labels==i

        if include_boundary:
            b = find_boundaries(m.astype(int),**kwargs)
            m = m+b

        #right now mask is in image convesion
        #if fature_value is false, convert
        if not feature_value:
            m = ~m
        
        output.append(m)

    return output



def masks_to_labels(masks,feature_value=False,**kwargs):
    """
    convert list of masks to labels

    Parameters
    ----------
    masks : list-like of masks

    feature value : bool (Default False)
        value which indicates feature.
        False is MaskedArray convension
        True is image convension

    Returns
    -------
    labels : array of labels
    """

    labels = np.zeros(masks[0].shape,dtype=int)

    for i,m in enumerate(masks):
        labels[m==feature_value] = i+1

    return labels
    


