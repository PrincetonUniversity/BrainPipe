#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:28:04 2018

@author: wanglab

"""
import h5py
from skimage.external import tifffile
import numpy as np
import os, time, sys
from os.path import basename, splitext, isfile
from tools.utils.io import load_kwargs
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
            
            
def h5_to_tiff(img_pth, img_tif_pth = False):
    '''Function to convert h5 files to tiff stacks. 
    Used for the 3dunet pipeline to check forward pass output.
    
    Inputs:
        img_pth = path to h5 file path that represents a numpy array/image
        
    Returns:
        img_tif_pth = tiff stack of h5 file
    '''
    #read file
    hf = h5py.File(img_pth, 'r+')    
    data = hf.get('main') #main should be the only key in 3dunet h5 files
    
    #make sure it is in numpy array format
    arr = np.array(data)
    
    #save file
    if img_tif_pth:
        tifffile.imsave(img_tif_pth, arr)
    else: 
        tifffile.imsave(img_pth[:-3]+'.tif', arr)
        
def fullsizedata_to_h5(dst, dct, memmap = True):
    '''Function to convert full_sizedatafld tiffs from a whole brain lightsheet image into a single h5 file to feed into 3dunet.
    Faster than convert tiffs to stack and stack to h5.
    Inputs:
        dct = processed data folder containing full_sizedatafld of cell channel or memory mapped array
        dst = destination path of h5 file
        memmap = False (default). If memmap already exists, specify path to write to .h5 file
    Returns:
        hdf5 dataset
    '''
    start = time.time()
    
    if not memmap:
        kwargs = load_kwargs(dct)
    
        #define cell channel volume
        vol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch'][0]
        
        #find all z planes in folder
        zplns = sorted(glob(os.path.join(vol.full_sizedatafld_vol, vol.brainname + '_ch*_C*_Z*.tif'))) 
        
        #setting constants and file name
        zpln = tifffile.TiffFile(zplns[0])
        h5_filename = os.path.join(dst, vol.brainname + '.h5')
    
        #create memory mapped array
        fp = load_memmap_arr(os.path.join(dst, vol.brainname + '.npy'), mode='w+', shape=(len(zplns), zpln.asarray().shape[0], zpln.asarray().shape[1]), dtype='uint16')
        
        #populate with each zpln
        for i, fn in enumerate(zplns):
            im = tifffile.TiffFile(fn)
            fp[i, :, :] = im.asarray() #load data onto memory mapped array
            
    else: #if memory mapped array exist, just load it
        h5_filename = dct[:-4]+'.h5'
        fp = load_memmap_arr(dct, mode='r+', dtype = 'uint16')
            
    #save out h5 file    
    with h5py.File(h5_filename) as h5_file:
        
        h5_file.create_dataset('/main', data = fp)
   
    #delete mmemp array - is this necessary???
    del fp
    
    sys.stdout.write('\n\nHDF5 file generated for {}\n       in {} minutes\n\n'.format(vol.brainname, (time.time() - start)/60))

def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''
    by: tpisano
    
    Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr         

#%%
if __name__ == "__main__":
   
   dct = '/jukebox/LightSheetTransfer/cnn/test/20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp.npy'
   dst = '/jukebox/LightSheetTransfer/cnn/test/'
   
   fullsizedata_to_h5(dst, dct, memmap = True)
   
