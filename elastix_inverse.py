
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
src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet"

param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"

mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/tiffs/WHS_SD_rat_T2star_v1.01_atlas.tif"
fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/z269_ch01_resized_forMRIr.tif"
outputdirectory = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/output_dirs/MRIr_to_z269_ch01"

# need to make moving larger (~140% seems to work well?) to transform to fixed
moving = tif.imread(mv)
fixed = tif.imread(fx)

if not os.path.exists(outputdirectory):
    os.mkdir(outputdirectory)

params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, mv, outputdirectory, params)

