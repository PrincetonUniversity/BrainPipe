
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 2021

working on aligning individual tiffs from histology stack first to each other, then to a substack of mPRA for Adrian

@author: ejdennis
"""
import os
import tifffile as tif
import sys
import numpy as np
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")
from tools.utils.directorydeterminer import set_src
[base_src,git_src,_git_analysis_src,src] = set_src("linux")
param_fld = os.path.join(git_src,"parameterfolder_rigid")
from tools.utils.io import makedir, removedir, writer, load_kwargs
import subprocess as sp


def elastix_command_line_call(fx, mv, out, parameters, resultnum):
    '''Wrapper Function to call elastix using the commandline, this can be time consuming

    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' noninverse transforms)
    out = folder to save file
    parameters = list of paths to parameter files IN ORDER THEY SHOULD BE APPLIED
    fx_mask= (optional) mask path if desired

    Outputs
    --------------
    ElastixResultFile = '.tif' or '.mhd' result file
    TransformParameterFile = file storing transform parameters

    '''
    e_params = ['elastix', '-f', fx, '-m', mv, '-out', out]

    # adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(parameters[x])
    writer(out, 'Elastix Command:\n{}\n...'.format(e_params))

    # set paths
    TransformParameterFile = os.path.join(
        out, 'TransformParameters.{}.txt'.format((len(parameters)-1)))
    ElastixResultFile = os.path.join(out,'{}_{}.tif'.format(str(resultnum),str(len(parameters)-1)))

    try:
        print('Running Elastix, this can take some time....\n')
        sp.call(e_params)  # sp_call(e_params)#
        writer(out, 'Past Elastix Commandline Call')
    except RuntimeError as e:
        writer(out, '\n***RUNTIME ERROR***: {} Elastix has failed. Most likely the two images are too dissimiliar.\n'.format(e.message))
        pass
    os.rename(os.path.join(out,'result.{}.tif'.format(str(len(parameters)-1))),ElastixResultFile)
    return ElastixResultFile, TransformParameterFile


# set your slices here -- assumes you've split your images into brightfield and fluorescent in FIJI
# I cropped one image, and made a folder within output_dirs called A243
mvtiffs = []
rn = -1
for i in np.arange(0,24):
	mvtiffs.append("fluorescent{0:0=2d}".format(i))
fxtiff = "fluorescent18"

fx = os.path.join(src,"tiffs/A243/{}.tif".format(fxtiff))


for mvtiff in mvtiffs:
	mv = os.path.join(src,"tiffs/A243/{}.tif".format(mvtiff))
	outputdirectory = os.path.join(src,"output_dirs/A243")

	if not os.path.exists(outputdirectory):
    		os.mkdir(outputdirectory)

	params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
	rn=rn+1
	e_out, transformfiles = elastix_command_line_call(fx, mv, outputdirectory, params,rn)
