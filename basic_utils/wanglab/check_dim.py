#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:10:47 2018

@author: wanglab
"""

import os, h5py
import numpy as np
from skimage import tifffile

#find all dimensions in the directory
src = '/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages'
for i, fn in enumerate(os.listdir(src)):
    f = h5py.File(os.path.join(src,fn))
    d = f["/main"].value
    f.close()
    print fn, d.shape, np.nonzero(d)[0].shape
    
#find dimensions of tif file
#src = '/home/wanglab/Documents/python/data/training_data/train/raw/20170116_tp_bl6_lob7_ml_08_647_010na_z7d5um_150msec_10povlp_ch00_C00_440-475_01_inputRawImages.tif'
src = '/home/wanglab/Documents/python/data/training_data/train/label/20170116_tp_bl6_lob7_ml_08_647_010na_z7d5um_150msec_10povlp_ch00_C00_440-475_01_inputLabelImages-segmentation.tif'
d = tifffile.imread(src)
print d.shape, np.nonzero(d)[0].shape
