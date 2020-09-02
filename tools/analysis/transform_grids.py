#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Jul 27 17:37:30 2020



@author: wanglab

"""


import os
import sys
import time

sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe")
import tifffile as tif
from tools.utils.io import makedir
from tools.registration.register import change_interpolation_order
from tools.registration.register import transformix_plus_command_line_call
from tools.registration.transform_list_of_points import modify_transform_files
from scipy.ndimage.interpolation import zoom


# setting paths
src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/"
ann = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/enlarged_tiffs/PMA_forPRA.tif"
# ann = os.path.join(src, "WHS_SD_rat_atlas_v3_annotation.tif")
fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/tiffs/median_image.tif"

dst = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/output_dirs/mouserat_affine_brain"
makedir(dst)

reg = os.path.join(src, "transform_files/mouserat_affine")
a2r = [os.path.join(reg, xx) for xx in os.listdir(reg) if "Transform" in xx]
a2r.sort()

# transformix
transformfiles = modify_transform_files(transformfiles=a2r, dst=dst)
[change_interpolation_order(xx, 0) for xx in transformfiles]

# change the parameter in the transform files that outputs 16bit images instead
for fl in transformfiles:  # Read in the file
    with open(fl, "r") as file:
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace(
        '(ResultImagePixelType "float")', '(ResultImagePixelType "short")')
    # Write the file out again
    with open(fl, "w") as file:
        file.write(filedata)

# run transformix
transformix_plus_command_line_call(ann, dst, transformfiles[-1])
