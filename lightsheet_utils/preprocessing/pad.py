#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:01:52 2018

@author: wanglab
"""

import tifffile as tif
import numpy as np
import time
import os

def pad(src, start):
    img = tif.imread(src).astype('float32')
    
    print(img.shape, img.dtype)
    
    imgshp = img.shape
    ptchsz = (20,192,192) #cnn window size
    
    dim = 2 #problem is with x
    
    if imgshp[dim] < ptchsz[dim]:
        pad = np.zeros((imgshp[0], imgshp[1], (ptchsz[dim]-1) - (imgshp[dim]-1)))
        img = np.append(img, pad, axis = dim)
    
    tif.imsave(src, img.astype('float32'), compress = 1)
    
    #sanity check
    img = tif.imread(src).astype('float32')
    print(img.shape, img.dtype)
          
    print('took {} secs'.format(round((time.time()-start),3)))    
    
if __name__ == '__main__':
    
    pth = '/home/wanglab/mounts/scratch/20180327_jg40_bl6_sim_03/patches'
    fls = os.listdir(pth); fls.sort()
    
    for i in range(32, len(fls), 3):
        src = os.path.join(pth, fls[i])
        start = time.time()    
        pad(src, start)