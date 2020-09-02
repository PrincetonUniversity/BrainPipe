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

# valdes to MRIr
ann = os.path.join(src, "valdes-hernandes_labels.tif")
fx = os.path.join(src, "WHS_SD_rat_T2star_v1.01_atlas.tif")

# run only on affine
# where are the parameter files
reg = os.path.join(src, "valdes_MRIr_affine")
a2r = [os.path.join(reg, xx) for xx in os.listdir(reg) if "Transform" in xx]
a2r.sort()

dst = os.path.join(src,"valdes_to_MRIr_affine")
makedir(dst)

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
transformix_plus_command_line_call(os.path.join(
    src, "valdes_annotations_MRIr.tif"),
    dst, transformfiles[-1])
