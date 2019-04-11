"""
routines to find maxima and segment and combine segments
"""

import numpy as np
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries



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
    if isinstance(structure,(bytes,str)) and structure=='set':
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
