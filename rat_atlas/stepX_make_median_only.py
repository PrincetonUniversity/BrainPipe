#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:49:34 2019

@author: wanglab
"""

import os, tifffile, cv2, numpy as np, multiprocessing as mp, sys, shutil, subprocess as sp
from scipy.ndimage import zoom

def listdirfull(x, keyword=False):
    """
    lists all contents of a directory by joining all paths
    """
    if not keyword:
        lst = [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx]
    else:
        lst = [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx and keyword in xx]

    lst.sort()
    return lst


def load_memmap_arr(pth, mode="r", dtype = "uint16", shape = False):
    """
    Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | "r"  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | "r+" | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | "w+" | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | "c"  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    """
    if shape:
        assert mode =="w+", "Do not pass a shape input into this function unless initializing a new array"
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

def generate_median_image(output_folder, memmappth, dst, verbose = True):
    """
    Function to collect post-registered volumes, generate a memory mapped array and then save out median volume
    """
    brains=os.listdir(output_fld)
    print(brains)
    vol = tifffile.imread(os.path.join(output_fld,brains[0]))
    z,y,x = vol.shape
    dtype = vol.dtype

    #init array
    arr = load_memmap_arr(memmappth, mode="w+", shape = (len(brains),z,y,x), dtype = dtype)

    #load
    for i, brain in enumerate(brains):
        arr[i] = tifffile.imread(os.path.join(output_fld,brain))
        arr.flush()
    if dst[-4:] != ".tif": dst = dst+".tif"
    if verbose: sys.stdout.write("...completed\nTaking median and saving as {}".format(dst)); sys.stdout.flush()

    #median volume
    vol = np.median(arr, axis=0)
    tifffile.imsave(dst, vol.astype(dtype))
    if verbose: sys.stdout.write("...completed"); sys.stdout.flush()
    return


if __name__ == "__main__":

    output_fld = "/home/emilyjanedennis/Desktop/mPMA"
    memmappth = os.path.join(output_fld, "mPMA_tom1seed.npy")
    #Location to save out our atlas (median image)
    final_output_path = os.path.join(output_fld, "mPMA_tom1seed.tif")

    #RUN AFTER ALL REGISTERATIONS ARE COMPLETE (locally or on head node, do not need a job for this)
    generate_median_image(output_fld, memmappth, final_output_path, verbose = True)
