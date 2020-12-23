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


mvtiffs = ["a235","c223","c514","c515","e106","f119","h170","k303"]
fxtiff = "PRA_25um"

fx = os.path.join(src,"tiffs/{}.tif".format(fxtiff))

for mvtiff in mvtiffs:
        mv = os.path.join(src,"enlarged_tiffs/{}.tif".format(mvtiff))
        moving_for_fixed = tif.imread(mv)
        reg = os.path.join(src,"output_dirs/{}_in_PRA".format(mvtiff))
        a2r = [os.path.join(reg,xx) for xx in os.listdir(reg) if "0_Transform" in xx]

        dst = os.path.join(src,"output_dirs/{}_in_PRA_affine".format(mvtiff))
        makedir(dst)

        # run transformix
        transformix_plus_command_line_call(mv, dst, a2r[-1])

#############################################################################
#############################################################################
