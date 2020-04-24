#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:08:27 2019

@author: wanglab
"""

import numpy as np, os, tifffile

src = "/home/wanglab/mounts/wang/pisano/tracing_output/retro_4x/20180313_jg_bl6f_prv_23/full_sizedatafld/20180313_jg23_4x_647_008na_1hfds_z7d5um_300msec_10povlp_ch00"

#trying to get hypothal
plns = list(range(620, 670))

zplns = [[os.path.join(src, xx) for xx in os.listdir(src) if "Z0{}".format(pln) in xx][0] for pln in plns]
zplns.sort()

assert len(zplns) == 50

y,x = tifffile.imread(zplns[0]).shape

arr = np.zeros((len(zplns), y, x))

for i,zpln in enumerate(zplns):
    arr[i] = tifffile.imread(zpln)
    
tifffile.imsave("/home/wanglab/Desktop/prv_23_z620-670_hypothal.tif", arr)
    
