#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019

@author: wanglab
"""

from tools.registration.register import elastix_command_line_call
import os
import tifffile as tif
import sys
from scipy.ndimage import zoom
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")

# k315 and k310 to MRIr

param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"

mv = "/home/emilyjanedennis/Desktop/brains/k315_1_1x_555_016na_1hfds_z10um_50msec_20povlp_resized_ch00.tif"
fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/tiffs/WHS_SD_rat_T2star_v1.01_atlas.tif"
outputfilename = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/k315_for_MRIr.tif"
outputdirectory = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/output_dirs/k315_to_MRIr"

if not os.path.exists(outputdirectory):
    os.mkdir(outputdirectory)

params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, outputfilename, outputdirectory, params)

