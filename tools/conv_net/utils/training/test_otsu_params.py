#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:58:26 2019

@author: wanglab
"""

import os, numpy as np, multiprocessing as mp
from skimage.external import tifffile
from skimage import filters
from utils.io import listdirfull, load_np, makedir

def otsu_par(saveLocation, otsufld, size, otsu_factor):
    """
    for inn in listdirfull(saveLocation, "npy"):
        print inn
        #load
        arr = load_np(inn)
        raw = np.copy(arr[0])
        lbl = np.copy(arr[1])
        
        #save input
        inrawfld = os.path.join(otsufld, "inputRawImages"); makedir(inrawfld)
        #[tifffile.imsave(os.path.join(inrawfld, "{}_{}_inputRawImages.tif".format(os.path.basename(inn)[:-4], str(i).zfill(4))), ii) for i, ii in enumerate(arr[0])]
        f = h5.File(os.path.join(inrawfld, "{}_inputRawImages.h5".format(os.path.basename(inn)[:-4])), "w")
        f["/main"] = raw
        f.close()
        
        #save input
        #OTSU?
        inlblfld = os.path.join(otsufld, "inputLabelImages"); makedir(inlblfld)
        otsu = otsu_dilate(raw, lbl, size=size, otsu_factor=otsu_factor).astype("float32")
        f = h5.File(os.path.join(inlblfld, "{}_inputLabelImages-segmentation.h5".format(os.path.basename(inn)[:-4])), "w")
        f["/main"] = otsu
        f.close()
    """
    #otsu
    p = mp.Pool(12)
    iterlst = [(otsufld, inn, size, otsu_factor) for inn in listdirfull(saveLocation, "npy")]
    p.starmap(otsu_helper, iterlst)
    p.terminate()
    return
            
def otsu_helper(otsufld, inn, size, otsu_factor):
    
    #load
    arr = load_np(inn)
    raw = np.copy(arr[0])
    lbl = np.copy(arr[1])
    
    #save input
    otsu = otsu_dilate(raw, lbl, size=size, otsu_factor=otsu_factor)
    merged = np.stack([raw, otsu, np.zeros_like(raw)], -1)
    tifffile.imsave(os.path.join(otsufld, "{}_otsu{}_size{}x{}x{}_lbl_overlay.tif".format(os.path.basename(inn)[:-4], otsu_factor, 
                                 size[0], size[1], size[2])), merged)

    print(inn)
    
    return

def otsu_dilate(arr0, arr1, size=(8,60,60), otsu_factor=0.8):
    """4d arr
    arr0=raw data
    arr1=points
    size=(z,y,x)
    otsu_factor - scaling of the otsu value, >1 is less stringent, <1 remove more pixels
    """
    #get points
    pnts = np.asarray(np.nonzero(arr1)).T.astype("int64")
    outarr = np.zeros_like(arr1)
    
    for pnt in pnts:
        #print pnt
        vol = np.copy(arr0[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]])*1.0
#        vol = filters.gaussian(vol, sigma = 0.4)
        v=filters.threshold_otsu(vol)/float(otsu_factor)
        vol[vol<v]=0
        vol[vol>=v]=1
        nvol = np.maximum(outarr[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]], vol)
        outarr[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]]=nvol
    
    return outarr
    
#%%
    
if __name__ == "__main__":
    
    #convert firs
    saveLocation = "/home/wanglab/Documents/cfos_inputs/memmap"; makedir(saveLocation) #test folder that contains memory mapped arrays will img + lbl points
    otsufld = "/home/wanglab/Documents/cfos_inputs/adaptive_thres"; makedir(otsufld) #output folder
    size = (5, 10, 10)
    otsu_factor = 0.8
    
    #otsu_par
    otsu_par(saveLocation, otsufld, size, otsu_factor)  
    