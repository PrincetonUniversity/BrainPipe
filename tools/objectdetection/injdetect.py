# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:58:01 2016

@author: wanglab
"""

from scipy.ndimage.filters import gaussian_filter as gfilt
from scipy.ndimage import label
import numpy as np
import sys, os
from collections import Counter
from tools.expression_mask.mask import make_mask
from tools.utils.io import makedir, load_kwargs
import tifffile
import SimpleITK as sitk


def find_site(im, thresh=10, filter_kernel=(5,5,5)):
    """Find a connected area of high intensity, using a basic filter + threshold + connected components approach
    
    by: bdeverett

    Parameters
    ----------
    img : np.ndarray
        3D stack in which to find site (technically need not be 3D, so long as filter parameter is adjusted accordingly)
    thresh: float
        threshold for site-of-interest intensity, in number of standard deviations above the mean
    filter_kernel: tuple
        kernel for filtering of image before thresholding
    
    Returns
    --------
    bool array of volume where coordinates where detected
    """
    if type(im) == str: im = tifffile.imread(im)

    filtered = gfilt(im, filter_kernel)
    thresholded = filtered > filtered.mean() + thresh*filtered.std() 
    labelled,nlab = label(thresholded)

    if nlab == 0:
        raise Exception('Site not detected, try a lower threshold?')
    elif nlab == 1:
        return labelled.astype(bool)
    else:
        sizes = [np.sum(labelled==i) for i in range(1,nlab+1)]
        return labelled == np.argmax(sizes)+1


def inj_detect_using_labels(threshold = .1, resampledforelastix = False, num_labels_to_keep=100, show = False, save = True, masking = False, **kwargs):
    '''Loads, thresholds and then applied scipy.ndimage.labels to find connected structures.
    
    Inputs
    -------
    threshold = % of maximum pixel intensity to make zero
    resampledforelastix= False:use downsized image (still fairly large); True: use 'resampledforelastix' image (smaller and thus faster)
    num_labels_to_keep= optional; take the top xx labels    
    show = optional, dispalys the threholded image and the image after scipy.ndimage.label has been applied    
    save = True: saves npy array; False = returns the numpy array
    masking = (optional) if True: using BD's masking algorithm *BEFORE* of thresholding
    
    Returns
    --------
    if save=False: returns thresholded numpy array
    if save=True: returns path to saved npfl
        
    '''
    sys.stdout.write('\n\nStarting Injection site detection, you need ~3x Memory of size of volume to run\n\n'); sys.stdout.flush()
    
    kwargs = load_kwargs(**kwargs)
        
    #load inj vol
    if not resampledforelastix: injvol = tifffile.imread([xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0].downsized_vol+'.tif')
    if resampledforelastix: injvol = tifffile.imread([xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0].downsized_vol+'_resampledforelastix.tif')
    injvolcp = injvol.copy()

    #bd's masking using regression    
    if masking:
        sys.stdout.write('\nStarting Masking Step...'); sys.stdout.flush()        
        #load reg vol
        if not resampledforelastix: regvol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0].downsized_vol+'.tif'
        if resampledforelastix: regvol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0].downsized_vol+'_resampledforelastix.tif'

        #masking regression
        maskfld = os.path.join(kwargs['outputdirectory'], 'injection', 'mask'); makedir(maskfld)
        injvol = make_mask(injvol, regvol, step=0.05, slope_thresh=0.4, init_window=200, out=maskfld, despeckle_kernel=5, imsave_kwargs={'compress': 5}, save_plots=True, verbose=True, **kwargs)
        sys.stdout.write('\n\nCompleted Masking Step'); sys.stdout.flush()
        
    #threshold bottom xx% of pixels
    injvol[injvol <= np.max(injvol)*threshold] = 0 
    
    
    #look for connected pixels        
    sys.stdout.write('\nLooking for connected pixels....'); sys.stdout.flush()
    lbl, numfeat=label(injvol)
    if show: sitk.Show(sitk.GetImageFromArray(injvol))
    if show: sitk.Show(sitk.GetImageFromArray(lbl))
    del injvol
    
    sys.stdout.write('\n      {} number of unique labels detected, if a large number, increase the threshold value'.format(numfeat)); sys.stdout.flush()
    
    #get pixelid, pixelcounts
    pxvl_num = Counter([lbl[tuple(xx)] for xx in np.argwhere(lbl>0)])
    [sys.stdout.write('\n{} pixels at value {}'.format(num, pxvl)) for pxvl, num in pxvl_num.iteritems()]; sys.stdout.flush()

    #format into list
    num_pxvl = [[num, pxvl] for pxvl, num in pxvl_num.iteritems()]

    sys.stdout.write('\nKeeping the {} largest label(s)'.format(num_labels_to_keep)); sys.stdout.flush()
    #remove smaller labels     
    num_pxvl.sort(reverse=True)
    rmlst=num_pxvl[num_labels_to_keep:]
    for n, pxvl in rmlst:
        lbl[lbl==pxvl] = 0
    [sys.stdout.write('\n    Kept {} of pixel id({})'.format(xx[0], xx[1])) for xx in num_pxvl[:num_labels_to_keep]]; sys.stdout.flush()

    #remove nonzero pixels from original (preserving their original values)        
    injvol = injvolcp * lbl.astype('bool'); del injvolcp
    
    #save out points:
    if save:    
        if not resampledforelastix: svpth = os.path.join(kwargs['outputdirectory'], 'injection', '{}labelskept_{}threshold_injectionpixels_downsized.tif').format(num_labels_to_keep, threshold)        
        if resampledforelastix: svpth = os.path.join(kwargs['outputdirectory'], 'injection','{}labelskept_{}threshold_injectionpixels_resampledforelastix.tif').format(num_labels_to_keep, threshold)
        tifffile.imsave(svpth,injvol)        
        return svpth
    else:            
        return injvol
        
if __name__ == '__main__':
    #injvol = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/bl6_crII/db_bl6_20160622_crII_52-5hr_01_badper/db_bl6_20160622_crII_52-5hr_01_badper_488w_561_z3um_200msec_1hfds_resized_ch01_resampledforelastix.tif')
    #regvol = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/bl6_crII/db_bl6_20160622_crII_52-5hr_01_badper/db_bl6_20160622_crII_52-5hr_01_badper_488w_561_z3um_200msec_1hfds_resized_ch00_resampledforelastix.tif')

    outdr = '/home/wanglab/wang/pisano/tracing_output/bl6_crII/db_bl6_20160622_crII_52-5hr_01_badper'
    kwargs=dict([('outputdirectory',outdr)])

    array = inj_detect_using_labels(threshold = .15, resampledforelastix = False, num_labels_to_keep=100, show = False, save = False, **kwargs)
