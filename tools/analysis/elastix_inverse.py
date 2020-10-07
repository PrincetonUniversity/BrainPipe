
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

#have MRIr to z265, z268_01 z269_01
src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet"
param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"
outputdirectory_base = os.path.join(src,"output_dirs/z269_ch01_to_z269_ch00")
mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/z269_ch01_resized_forMRIr.tif"
moving = tif.imread(mv)

brains = ["z268_ch00","z268_ch00"]

for brain in brains:
    # need to make moving larger (~140% seems to work well?) to transform to fixed
    fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/{}_forMRIr.tif".format(brain)
    fixed = tif.imread(fx)
    outputdirectory = "{}{}".format(outputdirectory_base,brain)
    if not os.path.exists(outputdirectory):
        os.mkdir(outputdirectory)

        params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

        e_out, transformfiles = elastix_command_line_call(fx, mv, outputdirectory, params)
        print("finished {}".format(fx))
