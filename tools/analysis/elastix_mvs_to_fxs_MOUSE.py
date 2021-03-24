
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

param_fld = "/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/parameterfolder_a1b"

# waxholm = "WHS_SD_rat_T2star_v1.01_atlas"


femtiffs = os.listdir('/home/emilyjanedennis/Desktop/TomData/probably_females')
masctiffs = os.listdir('/home/emilyjanedennis/Desktop/TomData/probably_males')

for pairnum in np.arange(9,len(femtiffs)):
	if pairnum%2 == 0:
		mvtiff = femtiffs[pairnum]
		fxtiff = masctiffs[pairnum]
		mv = os.path.join('/home/emilyjanedennis/Desktop/TomData/probably_females',mvtiff)
		fx = os.path.join('/home/emilyjanedennis/Desktop/TomData/probably_males',fxtiff)

	else:
		mvtiff = masctiffs[pairnum]
		fxtiff = femtiffs[pairnum]
		fx = os.path.join('/home/emilyjanedennis/Desktop/TomData/probably_females',fxtiff)
		mv = os.path.join('/home/emilyjanedennis/Desktop/TomData/probably_males',mvtiff)

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
	params.sort()
	e_out, transformfiles = elastix_command_line_call(fx, outputfilename, outputdirectory, params)
	os.rename(os.path.join(outputdirectory,'result.1.tif'),"/home/emilyjanedennis/Desktop/TomData/{}_in_{}.tif".format(mvtiff,fxtiff))

