#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:41:09 2018

@author: tpisano
"""

from tools.utils.io import makedir, chunkit, load_np, load_dictionary, save_kwargs, listdirfull, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer as dd, pth_update
import numpy as np, multiprocessing as mp,os, pandas as pd, gc,sys
from tools.conv_net.functions.bipartite import pairwise_distance_metrics_multiple_cutoffs
from tools.conv_net.apply_cnn.evaluate_cnn import compute_p_r_f1
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from itertools import product
from tools.conv_net.functions.sphericity import filter_sphericity
from tools.conv_net.functions.size_filter import size_filter
from skimage.external import tifffile
from tools.objectdetection.run_cnn import apply_cnn_to_folder
from tools.objectdetection.postprocess_cnn import load_tiff_folder, filter_size
from tools.objectdetection.evaluate_performance import compute_p_r_f1
from skimage.morphology import closing, ball


#size filter only:
sf = pd.read_csv('/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/size_filter/performance.csv')
sf = sf.sort_values('F1', ascending=False)
sf=sf[sf.cutoff<=50]
lower = sf.Size_cutoff[:1].tolist()[0]

#find peak max
df = pd.concat([pd.read_csv(xx) for xx in listdirfull('/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/param_sweep_2')])
df = df.sort_values('f1', ascending=False)

#rerun annotations (pth, src, dst)
run '/home/wanglab/wang/pisano/Python/lightsheet/tools/conv_net/input/input_data.py' '/home/wanglab/wang/pisano/conv_net/annotations/better_res/tp5' '/home/wanglab/wang/pisano/conv_net/annotations/better_res/memmap_arrays'

#group:
fld = '/home/wanglab/wang/pisano/conv_net/annotations/better_res/memmap_arrays'
optimization_group = ['20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y3800-4150_x2400-2750.npy','20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y2400-2750_x4500-4850.npy','20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1000-1350_x2050-2400.npy']
#ending in 1350 looks good. both 3450 have ventricles below.
h129_0 = ['20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y2050-2400_x3100-3450.npy','20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1350-1700_x3100-3450.npy','20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y4500-4850_x1000-1350.npy']
h129_1 = ['20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00_z200-400_y1000-1350_x2050-2400.npy','20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00_z200-400_y4500-4850_x3450-3800.npy','20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00_z200-400_y4150-4500_x3450-3800.npy'] #these look good
h129_2 = ['20170130_tp_bl6_sim_1750r_03_647_010na_1hfds_z7d5um_50msec_10povlp_ch00_z200-400_y2050-2400_x1350-1700.tif','20170130_tp_bl6_sim_1750r_03_647_010na_1hfds_z7d5um_50msec_10povlp_ch00_z200-400_y2400-2750_x3100-3450.tif','20170130_tp_bl6_sim_1750r_03_647_010na_1hfds_z7d5um_50msec_10povlp_ch00_z200-400_y2400-2750_x4500-4850.tif']
prv_0 = []
prv_1 = []
prv_2 = []

#%%
#run cnn on all, and size filter
cutoffs = [0.1, 1.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125]
dst = '/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/memmap_arrays_cnn'; makedir(dst)
dct={}
for ii,pth in enumerate(optimization_group+brain1+brain2+brain3+brain4):
    #load, save out, and run cnn, and filter size
    arr = load_np(os.path.join(fld, pth))[0]
    dst0 = os.path.join(dst, pth); makedir(dst0)
    [tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), im) for i,im in enumerate(arr)]
    #selem = ball(13)[13]
    #[tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), closing(im, selem)) for i,im in enumerate(arr)]
    apply_cnn_to_folder(src=dst0, dst=dst0+'_cnn_output')
    centers = filter_size(dst0+'_cnn_output', lower = lower)
    
    #load gt and run metrics
    gt = np.asarray(np.nonzero(load_np(os.path.join(fld, pth))[1])).T
    out = pairwise_distance_metrics_multiple_cutoffs(gt, centers, verbose=False, cutoffs = cutoffs); out = [[cutoffs[i]]+list(xx) for i,xx in enumerate(out)]
    dct[pth]=out
    
#condense    
df = pd.DataFrame(data=None, columns = ['path', 'Size_cutoff', 'cutoff', 'TP', 'FP', 'FN', 'P','R','F1'])
for pth, out in dct.iteritems():
    for i in out:    
        cutoff = i[0]
        tp = i[1]
        fp = i[2]
        fn = i[3]
        p,r,f1=compute_p_r_f1(tp=tp, fn=fn, fp=fp)
        df.loc[len(df)] = [pth, lower, cutoff, tp, fp, fn, p, r, f1]
tdf = df.sort_values('F1', ascending=False)
outpth='/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/cnn_performance_quantification'; #os.makedirs(outpth)
tdf.to_csv(outpth+'/performance_w_closing_13.csv')

tdf=pd.read_csv(outpth+'/performance.csv')
tdf=tdf[tdf.cutoff<=50]
for pth in tdf.path.unique():
    print ''
    print pth
    print tdf[tdf.path==pth][:3]
    print ''

#%%
#run cnn on all, and peak max function
cutoffs = [0.1, 1.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125]
dst = '/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/memmap_arrays_cnn'; makedir(dst)
dct={}
for ii,pth in enumerate(optimization_group+brain1+brain2+brain3+brain4):
    #load, save out, and run cnn, and filter size
    arr = load_np(os.path.join(fld, pth))[0]
    dst0 = os.path.join(dst, pth); makedir(dst0)
    #[tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), im) for i,im in enumerate(arr)]
    #apply_cnn_to_folder(src=dst0, dst=dst0+'_cnn_output')
    print pth
    from tools.objectdetection.postprocess_cnn import postprocess_cnn
    centers = postprocess_cnn(src = dst0+'_cnn_output')['centers']
    
    #load gt and run metrics
    gt = np.asarray(np.nonzero(load_np(os.path.join(fld, pth))[1])).T
    out = pairwise_distance_metrics_multiple_cutoffs(gt, centers, verbose=False, cutoffs = cutoffs); out = [[cutoffs[i]]+list(xx) for i,xx in enumerate(out)]
    dct[pth]=out
    
#condense    
df = pd.DataFrame(data=None, columns = ['path', 'cutoff', 'TP', 'FP', 'FN', 'P','R','F1'])
for pth, out in dct.iteritems():
    for i in out:    
        cutoff = i[0]
        tp = i[1]
        fp = i[2]
        fn = i[3]
        p,r,f1=compute_p_r_f1(tp=tp, fn=fn, fp=fp)
        df.loc[len(df)] = [pth, cutoff, tp, fp, fn, p, r, f1]
tdf = df.sort_values('F1', ascending=False)
outpth='/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/cnn_performance_quantification'; #os.makedirs(outpth)
tdf.to_csv(outpth+'/performance_cnn_postprocess_func.csv')

tdf=pd.read_csv(outpth+'/performance.csv')
tdf=tdf[tdf.cutoff<=50]
for pth in tdf.path.unique():
    print ''
    print pth
    print tdf[tdf.path==pth][:3]
    print ''

    







