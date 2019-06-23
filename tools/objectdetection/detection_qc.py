#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:47:07 2018

@author: tpisano
"""

import os, sys, numpy as np, time
from tools.utils.io import load_kwargs, save_dictionary, makedir, load_dictionary
from tools.objectdetection.run_cnn import apply_cnn_to_folder
from tools.objectdetection.random_forest import apply_classifier
from tools.utils.directorydeterminer import pth_update
from tools.objectdetection.postprocess_cnn import load_tiff_folder
from tools.conv_net.functions.dilation import dilate_with_element, ball

def cell_detect_qc_wrapper(src=False, **kwargs):
    '''Function to take in pth to brain folder, and output qc data
    
    src = destination of package (main folder)
    '''
    #load
    if src: kwargs.update(load_kwargs(src))
    if not src: kwargs.update(load_kwargs(**kwargs))

    #run for each cellch
    for cellch in [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch']:
        1
        #cell_detect(src = cellch.full_sizedatafld_vol, dst = os.path.join(os.path.dirname(os.path.dirname(cellch.full_sizedatafld_vol)), 'cells', os.path.basename(cellch.full_sizedatafld_vol)), **kwargs)
        
    return


def overlay_cells(src, center_intensity_radius, dilation_radius = 5, load_range = False):
    '''
    
    src = fullsizedata folder
    center_intensity_radius = path to center_intensity_radius.p file 
    
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr/full_sizedatafld/20161214_db_bl6_crii_l_53hr_647_010na_z7d5um_75msec_5POVLP_ch00'
    center_intensity_radius = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr/cells/20161214_db_bl6_crii_l_53hr_647_010na_z7d5um_75msec_5POVLP_ch00_centers_intensity_radius.p'
    load_range = (400,450)
    dilation_radius = 5
    
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/full_sizedatafld/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00'
    center_intensity_radius = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_centers_intensity_radius.p'
    
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/test'
    center_intensity_radius = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/test_out.p'
    load_range = (0,100)
    
    '''
    
    #load centers
    center_intensity_radius = load_dictionary(center_intensity_radius) if type(center_intensity_radius) == str else center_intensity_radius
    centers = sorted(center_intensity_radius.keys(), key=lambda x: x[0])
    
    #adjust and compensate for load range
    if load_range: centers = [tuple((c[0]-load_range[0], c[1], c[2])) for c in centers if c[0] in range(load_range[0], load_range[1])]
    
    #load
    vol = load_tiff_folder(src, threshold=0, load_range = load_range)
    vol0 = np.zeros_like(vol).astype('bool')
    for c in centers:
        vol0[c[0],c[1],c[2]]=True
    
    vol0 = dilate_with_element(vol0*1, ball(dilation_radius)).astype('uint8')
    #
    
    return vol, vol0