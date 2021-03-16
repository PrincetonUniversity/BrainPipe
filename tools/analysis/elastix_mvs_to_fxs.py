
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019

@author: wanglab
"""

import os
import numpy as np
import tifffile as tif
import sys
from scipy.ndimage import zoom
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")
from tools.registration.register import elastix_command_line_call
src = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet"

param_fld = "/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/parameterfolder_affine"

# waxholm = "WHS_SD_rat_T2star_v1.01_atlas"

mvtiffs = ["tom4_and_a235","e106_and_tom3"]
fxtiffs = ["k320_and_tom1","tom2_and_c514_3"]

for pairnum in np.arange(0,len(mvtiffs)):
	mvtiff = mvtiffs[pairnum]
	fxtiff = fxtiffs[pairnum]
	#mvtiff = "t107_a235_median_e106_f002_median_median"
	#fxtiff = "c514_f003_median_f110_k320_median_median"
	print(fxtiff)
	fx = os.path.join(src,"tiffs/{}.tif".format(fxtiff))
	mv = os.path.join(src,"tiffs/{}.tif".format(mvtiff))
	outputfilename = os.path.join(src,"enlarged_tiffs/{}_for_{}.tif".format(mvtiff,fxtiff))
	print(outputfilename)
	outputdirectory = os.path.join(src,"output_dirs/{}_to_{}".format(mvtiff,fxtiff))

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


	if not os.path.exists(outputdirectory):
    		os.mkdir(outputdirectory)

	params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

	e_out, transformfiles = elastix_command_line_call(fx, outputfilename, outputdirectory, params)
