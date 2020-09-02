#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019

@author: wanglab
"""

import os, tifffile as tif, sys
from scipy.ndimage import zoom
sys.path.append("/jukebox/wang/zahra/python/BrainPipe")
from tools.registration.register import elastix_command_line_call

mv = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif"

fx = "/jukebox/LightSheetData/brodyatlas/atlas/2019_meta_atlas/median_image.tif"

#need to make MRI atlas larger (~140% of atlas?) to transform to PRA
watl = tif.imread(mv)
pra = tif.imread(fx)
zf,yf,xf = (pra.shape[0]/watl.shape[0])*1.4, (pra.shape[1]/watl.shape[1])*1.4, (pra.shape[2]/watl.shape[2])*1.4
watl_for_pra = zoom(watl, (zf,yf,xf), order = 1)

tif.imsave("/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas_for_pra_reg.tif",
           watl_for_pra.astype("uint16"))

mv = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas_for_pra_reg.tif"

out = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/waxholm_to_pra"
if not os.path.exists(out): os.mkdir(out)

param_fld = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/rat_registration_parameter_folder"
params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)
