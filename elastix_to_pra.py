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

# chon to PMA

mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/Chon-Allen-brain.tif"
fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/PMA.tif"
outputfilename = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/Chon_for_PMA-reg.tif"
outputdirectory = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/Chon_to_PMA"
param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"

# need to make MRI atlas larger (~140% of atlas?) to transform to PRA
atlasbrain = tif.imread(mv)
pra = tif.imread(fx)
zf, yf, xf = (pra.shape[0]/atlasbrain.shape[0])*1.4, (pra.shape[1] /
                                                      atlasbrain.shape[1])*1.4, (pra.shape[2]/atlasbrain.shape[2])*1.4
atlasbrain_for_pra = zoom(atlasbrain, (zf, yf, xf), order=1)

tif.imsave(outputfilename, atlasbrain_for_pra.astype("uint16"))

if not os.path.exists(outputdirectory):
    os.mkdir(outputdirectory)

params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, outputfilename, out, params)
