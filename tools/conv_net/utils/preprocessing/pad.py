#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:01:52 2018

@author: wanglab
"""

import tifffile as tif
import numpy as np
import time

def pad(src, start):
    img = tif.imread(src).astype("float32")
    
    print(img.shape, img.dtype)
    
    imgshp = img.shape
    ptchsz = (20,192,192) #cnn window size

    if imgshp[0] < ptchsz[0]:
        pad = np.zeros(((ptchsz[0]-1) - (imgshp[0]-1), imgshp[1], imgshp[2]))
        img = np.append(img, pad, axis = 0)
    elif imgshp[1] < ptchsz[1]:
        pad = np.zeros((imgshp[0], (ptchsz[1]-1) - (imgshp[1]-1), imgshp[2]))
        img = np.append(img, pad, axis = 1)    
    elif imgshp[2] < ptchsz[2]:
        pad = np.zeros((imgshp[0], imgshp[1], (ptchsz[2]-1) - (imgshp[2]-1)))
        img = np.append(img, pad, axis = 2)
    
    tif.imsave(src, img.astype("float32"), compress = 1)
    
    #sanity check
    img = tif.imread(src).astype("float32")
    print(img.shape, img.dtype)
          
    print("took {} secs".format(round((time.time()-start),3)))    

#%%    
if __name__ == "__main__":
    
#    pth = "/home/wanglab/mounts/scratch/20180327_jg40_bl6_sim_03/patches"
#    fls = os.listdir(pth); fls.sort()
#    
#    for i in range(32, len(fls), 3):
#        src = os.path.join(pth, fls[i])
#        start = time.time()    
#        pad(src, start)
    
    src = "/jukebox/scratch/zmd/bad/20170308_tp_bl6_lob8_lat_05/input_chnks/patch_0000000095.tif"
    start = time.time() 
    pad(src, start)