#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:28:04 2018

@author: wanglab, tpisano
"""
import h5py
from skimage.external import tifffile
from skimage.io import imread
import multiprocessing as mp
import numpy as np
import os, time, sys, cv2
from math import ceil
from os.path import basename, splitext, isfile
from tools.utils.io import load_kwargs, writer
from glob import glob

#%%
def convert_tiff(tiff_filename):
    '''
    by James Gornet @ Columbia
    '''
    with tifffile.TiffFile(tiff_filename) as tif:
        if not isfile(tiff_filename):
            error_string = "{} does not exist".format(basename(tiff_filename))
            raise EnvironmentError(error_string)
        
        h5_filename = splitext(tiff_filename)[0] + ".h5"

        with h5py.File(h5_filename) as h5_file:
            h5_file.create_dataset("/main", data=tif.asarray())
            
            
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
        
def fullsizedata_to_h5(dct, dst):
    '''Function to convert full_sizedatafld tiffs from a whole brain lightsheet image into a single h5 file to feed into 3dunet.
    Faster than convert tiffs to stack and stack to h5.
    Inputs:
        dct = processed data folder containing full_sizedatafld of cell channel
        dst = destination path of h5 file
    
    Returns:
        hdf5 dataset
    Inspired by: https://stackoverflow.com/questions/31951507/out-of-core-4d-image-tif-storage-as-hdf5-python
    '''
    start = time.time()
    kwargs = load_kwargs(dct)
    #define cell channel volume
    vol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch'][0]
    #define full_sizedatafld subfolder containing cell channel tifs
    cellch_vol = vol.full_sizedatafld_vol
    zplns = sorted(glob(os.path.join(cellch_vol, vol.brainname + '_ch00_C00_Z0*.tif'))) #find all z planes in folder

    zpln = tifffile.TiffFile(zplns[0])
    h5_filename = os.path.join(dst, vol.brainname + '.h5')
    
    #create memory mapped array
    fp = np.memmap(os.path.join(dst, vol.brainname + '.dat'), dtype='uint16', mode='w+', shape = (len(zplns), zpln.asarray().shape[0], zpln.asarray().shape[1]))
    
    #populate h5 file with each zpln
    for i, fn in enumerate(zplns):
        im = tifffile.TiffFile(fn)
        fp[i, :, :] = im.asarray()
            
    #save out h5 file    
    with h5py.File(h5_filename) as h5_file:
        h5_file.create_dataset('/main', data = fp)
   
    del fp
    
    sys.stdout.write('\n\nHDF5 file generated for {}\n       in {} minutes\n\n'.format(vol.brainname, (time.time() - start)/60))
         