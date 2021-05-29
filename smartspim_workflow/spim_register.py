#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:01:17 2020

@author: wanglab
"""

import sys
import os
sys.path.append("/scratch/ejdennis/rat_BrainPipe")
from tools.registration.register import elastix_command_line_call

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
    cell = str(sys.argv[5])  # folder for cell channel e.g. Ex_642_Em_2
except:
    cell = False
try:
    species = str(sys.argv[6])  # species to know for registration parameters
    param_fld = "/scratch/ejdennis/rat_registration_parameter_folder"  # change if using mouse
except:
    print('nope')
    species = "rat"
    param_fld = "/scratch/ejdennis/rat_registration_parameter_folder"  # change if using mouse
try:
    atl = str(sys.argv[7])
except:
    atl = "/jukebox/brody/lightsheet/atlasdir/mPRA.tif"  # defaults to pra

if stepid == 0:
    print("step id is zero")
    svpth = os.path.join("/scratch/ejdennis/spimout", sys.argv[4])

    # path to store downsized images
    dst = os.path.join(svpth, "downsized")

    mv = os.path.join(dst, reg, "downsized_for_atlas.tif")
    print("\nPath to downsized vol for registration to atlas: %s" % mv)
    fx = atl
    print("\nPath to atlas: %s" % fx)
    out = os.path.join(os.path.dirname(src), "elastix")
    if not os.path.exists(out):
        os.mkdir(out)

    params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
    # run
    e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

    if cell:
        # cell vol to registration vol
        print("\nCell channel specified: %s" % cell)
        mv = os.path.join(src, cell+"/downsized_for_atlas.tif")
        fx = os.path.join(src, reg+"/downsized_for_atlas.tif")

        out = os.path.join(src, "elastix/%s_to_%s" % (cell, reg))
        if not os.path.exists(out):
            os.mkdir(out)

        params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
        # run
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

else:
    print("stepid is one")
    # atlas to registration vol
    # inverse transform
    fx = os.path.join(src, "reg__downsized_for_atlas_t.tif")
    mv = atl
    print("\nPath to downsized vol for inverse registration to atlas: %s" % fx)
    print("\nPath to atlas: %s" % mv)
    out = os.path.join(src, "elastix_inverse_transform")
    if not os.path.exists(out):
        os.mkdir(out)

    params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
    # run
    e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

