#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:01:17 2020

@author: wanglab
"""

import sys
import os
sys.path.append("../")
import tifffile as tif
import numpy as np
from tools.registration.register import elastix_command_line_call

param_fld = "/scratch/ejdennis/rat_registration_parameter_folder"  # change if using mouse
param_fld_affine = "/scratch/ejdennis/rat_BrainPipe/parameterfolder_affine"
atl = "/scratch/ejdennis/mPRA_adj_crop.tif"  # defaults to pra


# takes 6 command line arguments max
stepid = str(sys.argv[1])
print("stepid is 1 is {}".format(str(stepid==1)))
print("stepid is {}".format(stepid))
src = str(sys.argv[2])  # folder to main image folder
print("src is {}".format(src))

try:
    reg = str(sys.argv[3])  # folder fo registration channel, e.g. Ex_488_Em_0
except:
    print('no reg')
try:
    cell = str(sys.argv[4])  # folder for cell channel e.g. Ex_642_Em_2
except:
    cell = False

# sometimes these are one level up
if os.path.exists(os.path.join(src,'reg__downsized_for_atlas.tif')):
    output_src = src
elif os.path.exists(os.path.join(os.path.dirname(src),'reg__downsized_for_atlas.tif')):
    output_src = os.path.dirname(src)
else:
    output_src = os.path.dirname(os.path.dirname(src))

elsrc=os.path.join(os.path.dirname(src),"elastix")
print("elsrc is {}".format(elsrc))
if not os.path.exists(elsrc):
    os.mkdir(elsrc)

if stepid == 0:
    print("step id is zero")
    mv = os.path.join(output_src, "reg__downsized_for_atlas.tif")
    #print("\nPath to downsized vol for registration to atlas: %s" % mv)
    fx = atl

    
    if len(cell) > 1:
        # cell vol to registration vol
        print("\nCell channel specified: %s" % cell)
        mv = os.path.join(output_src, "cell__downsized_for_atlas.tif")
        fx = os.path.join(output_src, "reg__downsized_for_atlas.tif")
        out = os.path.join(elsrc, "cell_to_reg")
        if not os.path.exists(out):
            os.mkdir(out)

        params = [os.path.join(param_fld_affine, xx) for xx in os.listdir(param_fld_affine)]
        # run
        print("------------------- {} TO {} IN {} ------------------".format(mv,fx,out))
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

else:
    print("stepid is one")
    # atlas to registration vol
    # inverse transform

    if len(cell)>1:
        #cell to reg inverse
        fx=os.path.join(output_src,"cell__downsized_for_atlas.tif")
        mv=os.path.join(output_src, "reg__downsized_for_atlas.tif")
        out = os.path.join(elsrc, "reg_to_cell")
        if not os.path.exists(out):
            os.mkdir(out)
        
        params=[os.path.join(param_fld_affine, xx) for xx in os.listdir(param_fld_affine)]
        print("------------------- {} TO {} IN {} +++++++++++++++++++++".format(mv,fx,out))
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)
