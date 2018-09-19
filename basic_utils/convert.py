#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:28:04 2018

@author: wanglab
"""
import h5py
from skimage.external import tifffile

def convert_h5_to_tiff(img_pth, img_tif_pth = False):
    '''Function to convert h5 files to tiff stacks. 
    Used for the 3dunet pipeline to check forward pass output.
    
    Inputs:
        img_pth = path to h5 file path that represents a numpy array/image
        
    Returns:
        img_tif = tiff stack of h5 file
    '''
    #read file
    hf = h5py.File(img_pth, 'r')
    data = hf.get('main') #main should be the only key in the 3dunet h5 files
    #make sure it is in numpy array format
    arr = np.array(data)
    #save file
    if img_tif_pth:
        tifffile.imsave(img_tif_pth, arr)
    else: 
        tifffile.imsave('/home/wanglab/Desktop/test.tif', arr)