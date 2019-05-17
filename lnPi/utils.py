"""
utility functions
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import segmentation

def _convention_to_bool(convention):
    if convention == 'image':
        convention = True
    elif convention == 'masked':
        convention = False
    else:
        assert convention in [True, False]
    return convention


def mask_change_convention(mask,
                           convention_in='image',
                           convention_out='masked'):
    """
    convert an array from one convensiton to another

    convention : {'image', 'masked', True, False}

    if convention == 'image', values of True/False indicate inclusion/exclusion
    if convention == 'masksed', values of False/True indicate inclution/exclusion
    """
    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        mask = ~mask
    return mask


def masks_change_convention(masks,
                            convention_in='image',
                            convention_out='masked'):
    """
    convert convension of list of masks
    """

    convention_in = _convention_to_bool(convention_in)
    convention_out = _convention_to_bool(convention_out)

    if convention_in != convention_out:
        masks = [~m for m in masks]
    return masks


##################################################
# labels/masks utilities
##################################################
def labels_to_masks(labels,
                    features=None,
                    include_boundary=False,
                    convention='image',
                    check_features=True,
                    **kwargs):
    """
    convert labels array to list of masks

    Parameters
    ----------
    labels : array of labels to analyze

    features : array-like, optional
        list of features to extract from labels.  Note that returned
        mask[i] corresponds to labels == feature[i].
    include_boundary : bool (Default False)
        if True, include boundary regions in output mask
    convention : {'image','masked'} or bool.
        convention for output masks
    check_features : bool, default=True
        if True, and supply features, then make sure each feature is in labels

    **kwargs : arguments to find_boundary if include_boundary is True
        mode='outer', connectivity=labels.ndim

    Returns
    -------
    output : list of masks of same shape as labels
        mask for each feature
    features : list
        features

    """

    if include_boundary:
        kwargs = dict(dict(mode='outer', connectivity=labels.ndim), **kwargs)
    if features is None:
        features = [i for i in np.unique(labels) if i > 0]
    elif check_features:
        vals = np.unique(labels)
        assert np.all([x in vals for x in features])

    convention = _convention_to_bool(convention)

    output = []
    for i in features:
        m = labels == i
        if include_boundary:
            b = segmentation.find_boundaries(m.astype(int), **kwargs)
            m = m + b
        if not convention:
            m = ~m
        output.append(m)
    return output, features


def masks_to_labels(masks,
                    features=None,
                    convention='image',
                    dtype=np.int,
                    **kwargs):
    """
    convert list of masks to labels

    Parameters
    ----------
    masks : list-like of masks

    features : value for each feature, optional
        labels[mask[i]] = features[i] + feature_offset
        Default = range(1, len(masks) + 1)

    convention : {'image','masked'} or bool
        convention of masks

    Returns
    -------
    labels : array of labels
    """

    if features is None:
        features = range(1, len(masks) + 1)
    else:
        assert len(features) == len(masks)

    labels = np.full(masks[0].shape, fill_value=0,
                     dtype=dtype)

    masks = masks_change_convention(masks, convention, True)

    for i, m in zip(features, masks):
        labels[m] = i
    return labels






def ffill(arr, axis=-1, limit=None):
    import bottleneck
    _limit = limit if limit is not None else arr.shape[axis]
    return bottleneck.push(arr, n=_limit, axis=axis)


def bfill(arr, axis=-1, limit=None):
    '''inverse of ffill'''
    import bottleneck
    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    arr = np.flip(arr, axis=axis)
    # fill
    arr = bottleneck.push(arr, axis=axis, n=_limit)
    # reverse back to original
    return np.flip(arr, axis=axis)


##################################################
#calculations
##################################################


def get_mu_iter(mu, x):
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
            L.append(z + m)

    return np.array(L).T


##################################################
#utilities
##################################################


def sort_lnPis(input, comp=0):
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
