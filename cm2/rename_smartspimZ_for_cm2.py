#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created Jan 2021

@author: ejdennis
"""

import os,sys, glob,shutil
import numpy as np
import matplotlib.pyplot as plt

# symlink files to a new directory with filenames Z0000.tif, Z0001.tif, etc...
# Change the dst_dir to where you want these z planes linked 

src_dir = os.path.join('/jukebox/LightSheetData/lightserv/pbibawi',
			'pb_udisco_647_488_4x/pb_udisco_647_488_4x-003',
			'imaging_request_1/rawdata/resolution_3.6x',
			'Ex_642_Em_2/corrected/')

dst_dir = '/scratch/ejdennis/cm2_brains/e153/ch_642/renamed' 
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

src_files = sorted(glob.glob(src_dir + '/*tif'))
for ii,src in enumerate(src_files):
    dst_basename = 'Z' + f'{ii}'.zfill(4) + '.tif'
    dst = os.path.join(dst_dir,dst_basename)
    os.symlink(src,dst)
