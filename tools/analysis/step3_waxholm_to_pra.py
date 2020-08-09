# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:13:28 2019
@author: wanglab
"""

import os, tifffile as tif, sys
from scipy.ndimage import zoom
sys.path.append("/scratch/ejdennis/rat_BrainPipe")
from tools.registration.register import elastix_command_line_call

fx = "/jukebox/LightSheetData/brodyatlas/atlas/2019_meta_atlas/median_image.tif"
mv = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/schwarz_for_pra_reg.tif"

out = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/schwarz_to_pra"
if not os.path.exists(out): os.mkdir(out)

param_fld = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/rat_registration_parameter_folder"
params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]

e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)
