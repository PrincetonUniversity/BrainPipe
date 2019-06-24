#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:14:53 2017

@author: tpisano
"""
from __future__ import division
import numpy as np, os, sys
from skimage.external import tifffile
from tools.utils.io import listdirfull, load_kwargs
from tools.conv_net.input.read_roi import read_roi_zip
from tools.conv_net.functions.connected import detect_centers_from_contours
from tools.conv_net.input.mem_map import find_centers_from_contours
from scipy.ndimage.interpolation import zoom
import time
#%%
#Functions for resizing between the 1.3x objective and the 4x objective - could be useful for justifying/validating the lesser magnification
#resizes both input array as well as rois.

if __name__ == '__main__':
    impth = '/home/wanglab/wang/pisano/conv_net/training/3D/tom/db_20160616_cri_53hr_647_70msec_z3um_3hfds_C00_Z1245-1270/db_20160616_cri_53hr_647_70msec_z3um_3hfds_C00_Z1245-1270.tif'
    roipth =  '/home/wanglab/wang/pisano/conv_net/training/3D/tom/db_20160616_cri_53hr_647_70msec_z3um_3hfds_C00_Z1245-1270/db_20160616_cri_53hr_647_70msec_z3um_3hfds_C00_Z1245-1270.zip'
    src = '/home/wanglab/wang/pisano/conv_net/training/3D/temp.npy'
    dst = '/home/wanglab/wang/pisano/conv_net/training/3D/resized.npy'
    arr = find_centers_from_contours(impth, roipth, src, cores = 11, verbose=True)
    resize_array(src, in_resolution=(1.63,1.63,3), out_resolution=(5,5,3), dst=dst, remove_src = True, verbose = True)

#%%

def resize_array(src, in_resolution, out_resolution, dst, remove_src = False, verbose = False):
    '''Function to take in array generated from find_centers_from_contours or generate_mem_mapped_array_for_net_training and adjust size
    
    To load array after use:
        np.lib.format.open_memmap(pth_to_array, dtype = 'uint16', mode = 'r')
    
    
    Inputs:
    ------------------
    src: path of mem_mapped array generated from find_centers_from_contours or generate_mem_mapped_array_for_net_training (assumes 4D)
    in_resolution: tuple of x,y,z resolution of array
    out_resolution: tuple of x,y,z desired resolution of output array
    dst: output path of new resized mem_mapped array
    remove_src: (optional): if True remove original mem_mapped array(src)
    
    Returns:
    ------------------    
    dst: new path to array
    '''
    
    assert all((type(in_resolution)==tuple, type(out_resolution)==tuple)), 'in_resolution and out_resolution parameters need to be tuples. e.g.: in_resolution=(5,5,3)'
    
    #ensure floats
    in_resolution = tuple([float(xx) for xx in in_resolution])
    out_resolution = tuple([float(xx) for xx in out_resolution])
    
    #load
    arr = np.lib.format.open_memmap(src, dtype = 'uint16', mode = 'r+')
    #dims
    d,z,y,x = arr.shape
    
    #find scale factor for x,y,z
    scale = [xx[0]/xx[1] for xx in zip(in_resolution, out_resolution)]
    resize_ratio = [scale[2], scale[1], scale[0]] #zyx
    
    #pull out positive centers/contours and apply resize ratio
    assert arr.shape[0] < 3, 'Function as not been tested with two contours in the array (arr first dimension is >2).'
    centers = zip(*np.nonzero(arr[1,...]))

    #apply resize ratio
    centers = [[int(a*b) for a,b in zip(xx, resize_ratio)] for xx in centers]
    
    #resize
    if verbose: sys.stdout.write('\nStarting resizing of array...'); sys.stdout.flush()
    start = time.time()
    tarr = zoom(arr[0,...], (scale[2], scale[1], scale[0])); del arr
    narr = np.zeros((2, tarr.shape[0], tarr.shape[1], tarr.shape[2]))
    narr[0,...]=tarr

    #fill new centers
    for cen_z, cen_y, cen_x in centers:
        narr[1, cen_z, cen_y, cen_x] = 255
    
    if verbose: sys.stdout.write('...done. Completed in {} minutes'.format((time.time() - start) / 60)); sys.stdout.flush()
    
    #replace old mem_mapped location
    if verbose: sys.stdout.write('\n\nReinitializing memory mapped array:\n    {}...'.format(dst)); sys.stdout.flush()
    mapped_arr = np.lib.format.open_memmap(dst, dtype = 'uint16', mode = 'w+', shape = narr.shape)
    mapped_arr[:] = narr; mapped_arr.flush()
    if verbose: sys.stdout.write('...done.'); sys.stdout.flush()
    
    
    #optional remove src
    if remove_src: os.remove(src)
    
    return dst