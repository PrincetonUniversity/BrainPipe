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
atl = "/jukebox/brody/ejdennis/lightsheet/mPRA_adj.tif"  # defaults to pra


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

if stepid == 0:
    print("step id is zero")

    mv = os.path.join(dst, "reg__downsized_for_atlas.tif")
    mvtiff = tif.imread(mv)
    mvx,mvy,mvz = np.shape(mvtiff)
    if mvz > mvx:
        tif.imsave(mv,np.swapaxes(mvtiff,0,2))
        print("~~~~~~~~~~~~~~~ SWAPPED AXES ~~~~~~~~~~~~~")
    #print("\nPath to downsized vol for registration to atlas: %s" % mv)
    fx = atl
    print("\nPath to atlas: %s" % fx)
    elsrc=os.path.join(os.path.dirname(src),"elastix")
    print("elsrc is {}".format(elsrc))
    if not os.path.exists(elsrc):
        os.mkdir(elsrc)
    out = os.path.join(os.path.dirname(elsrc), "reg_to_atl")
    if not os.path.exists(out):
        os.mkdir(out)

    params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
    # run
    print("++++++++++++ {} TO {} IN {}+++++++++++".format(mv,fx,out))
    e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

    if cell:
        # cell vol to registration vol
        print("\nCell channel specified: %s" % cell)
        mv = os.path.join(src, "cell__downsized_for_atlas.tif")
        fx = os.path.join(src, "reg__downsized_for_atlas.tif")

        mvtiff = tif.imread(mv)
        mvx,mvy,mvz	= np.shape(mvtiff)
        if mvz > mvx:
            tif.imsave(mv,np.swapaxes(mvtiff,0,2))
            print("~~~~~~~~~~~~~~~ SWAPPED cell mv AXES	~~~~~~~~~~~~~")

        out = os.path.join(elsrc, "cell_to_reg")
        if not os.path.exists(out):
            os.mkdir(out)

        params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)][0]
        # run
        print("------------------- {} TO {} IN {} ------------------".format(mv,fx,out))
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

else:
    print("stepid is one")
    # atlas to registration vol
    # inverse transform
    fx = os.path.join(src, "reg__downsized_for_atlas.tif")
    mv = atl
    fxtiff = tif.imread(fx)
    fxx,fxy,fxz     = np.shape(fxtiff)
    if fxz > fxx: 
        tif.imsave(fx,np.swapaxes(fxtiff,0,2))
    print("\nPath to downsized vol for inverse registration to atlas: %s" % fx)
    print("\nPath to atlas: %s" % mv)
    out = os.path.join(src, "elastix_inverse_transform")
    if not os.path.exists(out):
        os.mkdir(out)

    params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
    # run
    print("------------------- {} TO {} IN {} +++++++++++++++++++++".format(mv,fx,out))
    e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

    if cell:
        #cell to reg inverse
        fx=os.path.join(src,"cell__downsized_for_atlas.tif")
        mv=os.path.join(src, "reg__downsized_for_atlas.tif")
        fxtiff = tif.imread(fx)
        fxx,fxy,fxz     = np.shape(fxtiff)
        if fxz > fxx:
            tif.imsave(fx,np.swapaxes(fxtiff,0,2))        
        out = os.path.join(src, "reg_to_cell")
        if not os.path.exists(out):
            os.mkdir(out)
        params=[os.path.join(param_fld, xx) for xx in os.listdir(param_fld)][0]
        print("------------------- {} TO {} IN {} +++++++++++++++++++++".format(mv,fx,out))
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)
