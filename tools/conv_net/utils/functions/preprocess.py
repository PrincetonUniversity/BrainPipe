#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:23:23 2018
@author: wanglab
by Tom Pisano (tpisano@princeton.edu, tjp77@gmail.com)
"""

import os, numpy as np, sys, multiprocessing as mp
from skimage.external import tifffile


def generate_patch(**params):
    """
    Function to patch up data and make into memory mapped array
    
    Inputs
    -----------
    src = folder containing tiffs
    patch_dst = location to save memmap array
    patchlist = list of patches generated from make_indices function
    stridesize = (90,90,30) - stride size in 3d ZYX
    patchsize = (180,180,60) - size of window ZYX
    mode = "folder" #"folder" = list of files where each patch is a file, "memmap" = 4D array of patches by Z by Y by X
    Returns
    ------------
    location of patched memory mapped array of shape (patches, patchsize_z, patchsize_y, patchsize_x)
    """
    #load array
    input_arr = load_memmap_arr(os.path.join(params["data_dir"], "input_memmap_array.npy"))
    
    #set patch destination
    patch_dst = os.path.join(params["data_dir"], "input_chnks")    
    
    #set variables
    if patch_dst[-4:]==".npy": patch_dst = patch_dst[:-4]
    if not os.path.exists(patch_dst): os.mkdir(patch_dst)
    window = params["window"]
    patchsize = params["patchsz"]
    
    jobid = int(params["jobid"]) #set patch no. to run through cnn
    #select the file to process for this array job
    if jobid > len(params["patchlist"])-1:
        sys.stdout.write("\njobid {} > number of files".format(jobid)); sys.stdout.flush()  
    else:
        #patch
        for i,p in enumerate(params["patchlist"]):
            if i == jobid: 
                v = input_arr[p[0]:p[0]+patchsize[0], p[1]:p[1]+patchsize[1], p[2]:p[2]+patchsize[2]]
                #padding to prevent cnn erros 
                if v.shape[0] < window[0]:
                    pad = np.zeros((window[0]-v.shape[0], v.shape[1], v.shape[2]))
                    v = np.append(v, pad, axis = 0)
                if v.shape[1] < window[1]:
                    pad = np.zeros((v.shape[0], window[1]-v.shape[1], v.shape[2]))
                    v = np.append(v, pad, axis = 1)
                if v.shape[2] < window[2]:
                    pad = np.zeros((v.shape[0], v.shape[1], window[2]-v.shape[2]))
                    v = np.append(v, pad, axis = 2)
                #saving out
                tifffile.imsave(os.path.join(patch_dst, "patch_{}.tif".format(str(i).zfill(10))), v.astype("float32"), compress=1)
                if params["verbose"]: print("{} of {}".format(i, len(params["patchlist"])))
    #return
    return patch_dst
   
    
def get_dims_from_folder(src):    
    """
    Function to get dims from folder (src)
    """
    
    fls = listdirfull(src, keyword = ".tif")
    y,x = tifffile.imread(fls[0]).shape
    return (len(fls),y,x)
    
def make_indices(inputshape, stridesize):
    """
    Function to collect indices
    inputshape = (500,500,500)
    stridesize = (90,90,30)
    """    
    zi, yi, xi = inputshape
    zs, ys, xs = stridesize
    
    lst = []
    z = 0; y = 0; x = 0
    while z<zi:
        while y<yi:
            while x<xi:
                lst.append((z,y,x))
                x+=xs
            x=0
            y+=ys
        x=0
        y=0
        z+=zs
    return lst

def make_memmap_from_tiff_list(src, dst, cores=8, dtype="float32", verbose=True):
    """
    Function to make a memory mapped array from a list of tiffs
    """

    if type(src) == str and os.path.isdir(src): 
        src = listdirfull(src, keyword = ".tif")
        src.sort()
    im = tifffile.imread(src[0])
    if not dtype: dtype = im.dtype
    
    #init
    dst = os.path.join(dst, "input_memmap_array.npy")
    memmap=load_memmap_arr(dst, mode="w+", dtype=dtype, shape=tuple([len(src)]+list(im.shape)))
    
    #run
    if cores<=1:
        for i,s in enumerate(src):
            memmap[i,...] = tifffile.imread(s)
            memmap.flush()
    else:
        iterlst = [(i,s, dst, verbose) for i,s in enumerate(src)]    
        p = mp.Pool(cores)
        p.starmap(make_memmap_from_tiff_list_helper, iterlst)
        p.terminate

    return dst

def make_memmap_from_tiff_list_helper(i, s, memmap_pth, verbose):
    """
    """
    #load
    arr = load_memmap_arr(memmap_pth, mode="r+")
    arr[i,...] = tifffile.imread(s)
    arr.flush(); del arr
    if verbose: sys.stdout.write("\ncompleted plane {}".format(i)); sys.stdout.flush()
    return

def listdirfull(x, keyword=False):
    """ 
    might need to modify based on server...i.e. if automatically saving a file called "thumbs"
    """
    if not keyword:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx]
    else:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx and keyword in xx]

def load_memmap_arr(pth, mode="r", dtype = "float32", shape = False):
    """
    Function to load memmaped array.
    
    by @tpisano
    """
    if shape:
        assert mode =="w+", "Do not pass a shape input into this function unless initializing a new array"
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr
    
def reconstruct_memmap_array_from_tif_dir(**params):
    """
    Function to take CNN probablity map tifs (patches, patchsize_z, patchsize_y, patchsize_x) and build into single 3d volume
    
    Inputs
    ---------------
    src = cnn_memory_mapped array of shape (patches, patchsize_z, patchsize_y, patchsize_x)
    recon_dst = path to generate numpy array
    inputshape = (Z,Y,X) shape of original input array
    patchlist = list of patches generated from make_indices function
    stridesize = (90,90,30) - stride size in 3d ZYX
    patchsize = (180,180,60) - size of window ZYX
    
    Returns
    ------------
    location of memory mapped array of inputshape
    """
    
    #load
    cnn_fls = os.listdir(params["cnn_dir"]); cnn_fls.sort()
    
    jobid = int(params["jobid"]) #set patch no. to run through cnn
    
    #grab array to read and write
    recon_array = load_memmap_arr(params["reconstr_arr"], mode="r+")
    
    #find patchlist
    patchlist = params["patchlist"]
    
    #iterate
    #select the file to process for this array job
    if jobid > len(params["patchlist"])-1:
        sys.stdout.write("\njobid {} > number of files".format(jobid)); sys.stdout.flush()  
    else:
        #patch
        for i,p in enumerate(patchlist):
            if i == jobid: 
                b = tifffile.imread(os.path.join(params["cnn_dir"], cnn_fls[i])).astype(params["dtype"])
                a = recon_array[p[0]:p[0]+b.shape[0], p[1]:p[1]+b.shape[1], p[2]:p[2]+b.shape[2]]
                if not a.shape == b.shape: b = b[:a.shape[0], :a.shape[1], :a.shape[2]]
                nvol = np.maximum(a,b)
                recon_array[p[0]:p[0]+b.shape[0], p[1]:p[1]+b.shape[1], p[2]:p[2]+b.shape[2]] = nvol
                recon_array.flush(); del b
                if params["verbose"]: print("{} of {}".format(i, len(patchlist)))
        
    sys.stdout.write("\nfinished reconstruction :]\nshape after reconstruction is: {}".format(recon_array.shape)); sys.stdout.flush()
    
    return params["reconstr_arr"]
