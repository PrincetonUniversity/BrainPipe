#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019

@author: wanglab
"""

import os, tifffile as tif, sys
from scipy.ndimage import zoom
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")
from tools.registration.register import elastix_command_line_call

mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/median_image.tif"
fx = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/PMA.tif"

#need to make MRI atlas larger (~140% of atlas?) to transform to PRA
atlasbrain = tif.imread(mv)
pra = tif.imread(fx)
zf,yf,xf = (pra.shape[0]/atlasbrain.shape[0])*1.4, (pra.shape[1]/atlasbrain.shape[1])*1.4, (pra.shape[2]/atlasbrain.shape[2])*1.4
atlasbrain_for_pra = zoom(atlasbrain, (zf,yf,xf), order = 1)

tif.imsave("/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/PRA_forPMA.tif",
           atlasbrain_for_pra.astype("uint16"))

mv = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/PRA_forPMA.tif"
out = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/ratmouse"
if not os.path.exists(out): os.mkdir(out)

param_fld = "/home/emilyjanedennis/Desktop/brains/w122/parameterfolder"
params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)
