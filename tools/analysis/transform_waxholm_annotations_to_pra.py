#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Jul 27 17:37:30 2020



@author: wanglab

"""


import os
import sys
import time

import tifffile as tif

#  sys.path.append("/jukebox/wang/zahra/python/BrainPipe")

from tools.utils.io import makedir

from tools.registration.register import change_interpolation_order
from tools.registration.register import transformix_command_line_call

from tools.registration.transform_list_of_points import modify_transform_files

from scipy.ndimage.interpolation import zoom


# setting paths

src = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/"

ann = os.path.join(src, "WHS_SD_rat_atlas_v3_annotation.tif")

fx = "/jukebox/LightSheetData/brodyatlas/atlas/2019_meta_atlas/median_image.tif"

# need to make MRI annotation larger (~140% of atlas?) to transform to PRA

watl = tif.imread(ann)

pra = tif.imread(fx)

zf, yf, xf = (pra.shape[0]/watl.shape[0])*1.4, (
    pra.shape[1] /
    watl.shape[1])*1.4, (pra.shape[2]/watl.shape[2])*1.4

print("\nzooming...")

watl_for_pra = zoom(watl, (zf, yf, xf), order=1)


# saved out annotation volume

print("\nsaving zoomed volume...")

tif.imsave(os.path.join(src, "WHS_SD_rat_atlas_v3_annotation_for_pra_reg.tif"),

           watl_for_pra.astype("uint16"))


reg = os.path.join(src, "waxholm_to_pra")

a2r = [os.path.join(reg, xx) for xx in os.listdir(reg) if "Transform" in xx]
a2r.sort()


dst = os.path.join(src, "transformed_annotation_volume")

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

transformix_command_line_call(os.path.join(
    src, "WHS_SD_rat_atlas_v3_annotation_for_pra_reg.tif"),

    dst, transformfiles[-1])
