#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:15:21 2019

@author: wanglab
"""

#in python 3!!

import os, numpy as np, multiprocessing as mp
from skimage.external import tifffile
from skimage import filters
from utils.io import listdirfull

def otsu_par(saveLocation, otsufld, guassian_sigma, otsu_factor):
   
    #otsu
    p = mp.Pool(12)
    iterlst = [(otsufld, inn, guassian_sigma, otsu_factor) for inn in listdirfull(saveLocation, "npy")]
    p.starmap(otsu_helper, iterlst)
    p.terminate()
    return
            
def otsu_helper(otsufld, inn, guassian_sigma, otsu_factor):
    
    #load
    arr = np.load(inn)
    raw = np.copy(arr[0])
    lbl = np.copy(arr[1])
    
    #save input
    otsu = otsu_dilate(raw, lbl, sigma = guassian_sigma, otsu_factor=otsu_factor)
    merged = np.stack([raw, otsu, np.zeros_like(raw)], -1)
    tifffile.imsave(os.path.join(otsufld, "{}_otsu{}_sigma{}_lbl_overlay.tif".format(os.path.basename(inn)[:-4], otsu_factor, 
                                 guassian_sigma)), merged)

    print(inn)
    
    return

def otsu_dilate(arr0, arr1, sigma, otsu_factor=0.8):
    """4d arr
    arr0=raw data
    arr1=points
    size=(z,y,x)
    otsu_factor - scaling of the otsu value, >1 is less stringent, <1 remove more pixels
    """
    vol = filters.gaussian(arr1, sigma = sigma)
    v = filters.threshold_otsu(vol)/float(otsu_factor)
    vol[vol < v] = 0
    vol[vol >= v] = 1
    
    return vol.astype("uint16")
    
    
if __name__ == "__main__":
    
    #convert first
    saveLocation = "/home/wanglab/Documents/cfos_inputs/memmap"
    if not os.path.exists(saveLocation): os.mkdir(saveLocation) #test folder that contains memory mapped arrays will img + lbl points
    thresfld = "/home/wanglab/Documents/cfos_inputs/otsu_and_guassian"
    if not os.path.exists(thresfld): os.mkdir(thresfld) #output folder
    otsu_factor = 4
    guassian_sigma = 1
    
    #otsu_par
    otsu_par(saveLocation, thresfld, guassian_sigma, otsu_factor)  