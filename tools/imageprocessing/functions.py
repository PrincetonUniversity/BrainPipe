#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:18:34 2017

@author: tpisano
"""
import cv2, SimpleITK as sitk
from skimage.external import tifffile
from numba import jit
from skimage.exposure import rescale_intensity
from skimage.morphology import erosion, dilation, disk
import numpy as np

#%%
if __name__ == "__main__":
    im = '/home/wanglab/LightSheetData/marm_ghazanfar/marm_standard_brain_atlas/T1_single_brain_DV_upsampled_sharpened.tif'
    #sitk.Show(sitk.GetImageFromArray(tifffile.imread(im)))
    im = clahe(im)
    #sitk.Show(sitk.GetImageFromArray(im))
    tifffile.imsave('/home/wanglab/LightSheetData/marm_ghazanfar/marm_standard_brain_atlas/T1_single_brain_DV_upsampled_sharpened_clahe.tif', im)

    im = '/home/wanglab/LightSheetData/marm_ghazanfar/marm_standard_brain_atlas/T2_single_brain_DV_upsampled_sharpened.tif'
    #sitk.Show(sitk.GetImageFromArray(tifffile.imread(im)))
    im = clahe(im)
    #sitk.Show(sitk.GetImageFromArray(im))
    tifffile.imsave('/home/wanglab/LightSheetData/marm_ghazanfar/marm_standard_brain_atlas/T2_single_brain_DV_upsampled_sharpened_clahe.tif', im)


#%%
def clahe(im, func = 'cv2'):
    '''Function to clahe image using cv2's clahe filter. 
    Note: cv2 filter only works on 8bit images so non 8bit images are converted to 8 bit, clahed, and then recoverted back.
    
    Inputs:
    -----------------
    im: pth to tiffile or numpy array to be clahed
    func = 'cv2' or 'skimage'
    
    Returns:
    ------------------
    array: numpy array
    '''
    
    #inputs:
    if type(im) == str: im = tifffile.imread(im)
    
    #im type:
    imtype = str(im.dtype)
    
    if func == 'cv2':
        
        print ('Using cv2, which only works on 8bit images, so losing data, change "func" input if necessary')
    
        if imtype != 'uint8': im = rescale_intensity(im, in_range=imtype, out_range='uint8').astype('uint8')
        
        imfilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if len(im.shape) ==2:
            im = imfilter.apply(im)
        elif len(im.shape)==3 and im.shape[2]==3:
            for i in range(3):
                    im[...,i] = imfilter.apply(im[...,i])
        else:
            for i in range(len(im)):
                    im[i,...] = imfilter.apply(im[i,...])
        
        #rescale
        im = rescale_intensity(im, in_range='uint8', out_range=imtype).astype(imtype)
    
    else:
        from skimage.exposure import equalize_adapthist
        if im.ndim == 3:im = np.asarray([equalize_adapthist(i, nbins = 512) for i in im]).astype(imtype)
        if im.ndim == 2: im = equalize_adapthist(im, nbins = 512).astype(imtype)
        
    return im


def norm(vol):
    try:
        nvol = (vol - np.min(vol)) / (float(np.max(vol)) - float(np.min(vol)))
    except Exception, e:
        print e
        nvol = vol
    return nvol

@jit
def erode_and_dilate(max_d, med_d, depthkernel, vol):
    '''max_d=1
    med_d = 30
    depthkernel = 5
    vol 
    '''
    d1 = disk(max_d) #this does do something
    d2 = disk(med_d) #30 is good, 50 seems better, especially for dense label
    nvol = np.zeros_like(vol)
    dk = int(depthkernel / 2)
    
    for i in range(len(vol)):
        nvol[i] = cv2.subtract(cv2.dilate(cv2.erode(np.max(vol[np.max((i-dk, 0)) : i+dk], axis=0), d1, iterations = 1), d1, iterations = 1).astype('float64'), cv2.dilate(cv2.erode(np.median(vol[np.max((i-dk, 0)) : i+dk], axis=0), d2, iterations = 1), d2, iterations = 1))    
    
    
    return nvol










