#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:23:13 2018

@author: tpisano
"""

from tools.utils.io import makedir, chunkit, load_np, load_dictionary, listdirfull
import numpy as np, os
from tools.utils.io import make_memmap_from_tiff_list
from skimage.util import view_as_windows
from skimage.external import tifffile


if __name__ == '__main__':
    #make 200,350,350
    zrng = range(200, 600, 200) #at least a delta of 100
    yrng = range(1000, 5000, 350)
    xrng = range(1000, 5000, 350)
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170204_tp_bl6_cri_1000r_02/full_sizedatafld/20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00'
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170116_tp_bl6_lob7_500r_09/full_sizedatafld/20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00' 
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170130_tp_bl6_sim_1750r_03/full_sizedatafld/20170130_tp_bl6_sim_1750r_03_647_010na_1hfds_z7d5um_50msec_10povlp_ch00'
    src = '/home/wanglab/wang/pisano/tracing_output/retro_4x/20180215_jg_bl6f_prv_10/full_sizedatafld/20180215_jg_bl6f_prv_10_647_010na_z7d5um_250msec_10povlap_ch00'
    src = '/home/wanglab/wang/pisano/tracing_output/retro_4x/20180215_jg_bl6f_prv_07/full_sizedatafld/20180215_jg_bl6f_prv_07_647_010na_z7d5um_250msec_10povlap_ch00'
    dst = os.path.join('/home/wanglab/Downloads/', os.path.basename(src)); makedir(dst)
    lst = listdirfull(src, keyword='.tif'); lst.sort()
    make_memmap_from_tiff_list(lst, dst+'.npy')
    
    arr = load_np(dst+'.npy'); makedir(dst)
    dst = os.path.join('/home/wanglab/wang/pisano/conv_net/annotations/better_res', os.path.basename(src)); makedir(dst)
    for i in range(len(zrng)-1):
        for ii in range(len(yrng)-1):
            for iii in range(len(xrng)-1):
                z, zz = zrng[i], zrng[i+1]
                y, yy = yrng[ii], yrng[ii+1]
                x, xx = xrng[iii], xrng[iii+1]
                tifffile.imsave(os.path.join(dst,'{}_z{}-{}_y{}-{}_x{}-{}.tif'.format(os.path.basename(src), z,zz,y,yy,x,xx)), arr[z:zz,y:yy,x:xx], compress=1)
    
    #don't typically need np arrays but just in case
    dst = os.path.join('/home/wanglab/wang/pisano/conv_net/annotations/better_res', os.path.basename(src)); makedir(dst)
    for i in range(len(zrng)-1):
        for ii in range(len(yrng)-1):
            for iii in range(len(xrng)-1):
                z, zz = zrng[i], zrng[i+1]
                y, yy = yrng[ii], yrng[ii+1]
                x, xx = xrng[iii], xrng[iii+1]
                np.save(os.path.join(dst,'{}_z{}-{}_y{}-{}_x{}-{}.tif'.format(os.path.basename(src), z,zz,y,yy,x,xx)), arr[z:zz,y:yy,x:xx])
    