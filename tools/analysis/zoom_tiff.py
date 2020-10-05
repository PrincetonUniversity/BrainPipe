
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019

@author: wanglab
"""

import os
import tifffile as tif
import sys
from scipy.ndimage import zoom
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")
from tools.registration.register import elastix_command_line_call

src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet"
param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"

fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/tiffs/WHS_SD_rat_T2star_v1.01_atlas.tif"
mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/tiffs/z269_0_resampled.tif"
outputfilename = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/z269_ch00_forMRIr.tif"

# need to make moving larger (~140% seems to work well?) to transform to fixed
moving = tif.imread(mv)
fixed = tif.imread(fx)
zf, yf, xf = (fixed.shape[0]/moving.shape[0])*1.4, (
    fixed.shape[1] /
    moving.shape[1])*1.4, (fixed.shape[2]/moving.shape[2])*1.4
print("\nzooming...")
moving_for_fixed = zoom(moving, (zf, yf, xf), order=0,mode='nearest')

# saved out volume
print("\nsaving zoomed volume...")
tif.imsave(outputfilename,moving_for_fixed.astype("uint16"))
