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
import numpy as np

# setting paths
src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/"
# ann = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/10grid.tif"

mvlist = ["k320","k321","k323","k327","t107","f002","f003","f110","k320","k321","k323","k327","t107","f002","f003","f110"]
fxlist = ["PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAf_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4","PRAm_seed4"]
#alignedlist =  ["WHS_SD_rat_T2star_v1.01_atlas","Chon-Allen-brain","PMA"]
alignedlist = mvlist

for i in np.arange(0,len(mvlist)):
	# setting paths
	mvname = mvlist[i]
	fxname = fxlist[i]
	alignedname = alignedlist[i]
	fx = os.path.join(src, "tiffs/{}.tif".format(fxname))
	mv = os.path.join(src, "tiffs/{}.tif".format(mvname))
	enlargedfilename= os.path.join(src, "enlarged_tiffs/{}_for_{}.tif".format(mvname,fxname))

	dst = os.path.join(src,"output_dirs/{}_in_{}".format(mvname,fxname))
	makedir(dst)
	transformfilepath = os.path.join(src, "output_dirs/{}_to_{}".format(alignedname,fxname))
	moving = tif.imread(mv)
	fixed = tif.imread(fx)

	# copy the parameter files
	a2r = [os.path.join(transformfilepath, xx) for xx in os.listdir(transformfilepath) if "Transform" in xx]
	
	a2r.sort()
	
	transformfiles = modify_transform_files(transformfiles=a2r, dst=dst)
	[change_interpolation_order(xx, 0) for xx in transformfiles]
	
	# change the parameter in the transform files that outputs 16bit images i>
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
	transformix_plus_command_line_call(mv, dst, transformfiles[-1])

