#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 20:41:01 2018

edited for rats by Emily Jane Dennis July 2020
@author: tpisano
"""

from ClearMap.cluster.utils import load_kwargs
from ClearMap.cluster.par_tools import celldetection_operations
from ClearMap.cluster.directorydeterminer import directorydeterminer
from ClearMap.cluster.preprocessing import updateparams, listdirfull, arrayjob, makedir, removedir
from skimage.exposure import rescale_intensity
import os
import sys
import shutil
import pickle
import tifffile
import numpy as np
from itertools import product
from xvfbwrapper import Xvfb
vdisplay = Xvfb()
vdisplay.start()

systemdirectory = directorydeterminer()
# set paths to data
# inputdictionary stucture: key=pathtodata value=list["xx", "##"]
# where xx=regch, injch, or cellch and ##=two digit channel number
# "regch" = channel to be used for registration, assumption is all
# other channels are signal
# "cellch" = channel(s) to apply cell detection
# "injch" = channels(s) to quantify injection site
# "##" = when taking a multi channel scan following regexpression,
# the channel corresponding to the reg/cell/inj channel.
# i.e. name_of_scan_channel00_Z#### then use "00"
# e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"],
# ["injch", "01"]]} ###create this dictionary variable BEFORE params

inputdictionary = {
    os.path.join(systemdirectory, "LightSheetTransfer/brody/z265"): [["regch", "00"]],
    os.path.join(systemdirectory, "LightSheetTransfer/brody/z265"): [["cellch", "00"]]
}

# Required inputs
params = {
    "inputdictionary": inputdictionary,  # don"t need to touch
    "outputdirectory": os.path.join(systemdirectory, "LightSheetData/rat-brody/processed/201910_tracing/clearmap/z265"),
    # False/None, float(e.g: 0.4), amount to resize by: >1 means increase size, <1 means decrease
    "resample": False,
    # micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
    "xyz_scale": (1.63, 1.63, 10.0),
    "tiling_overlap": 0.00,  # percent overlap taken during tiling
    # it is assumed that input image will be a horizontal scan with anterior being "up"; USE .TIF!!!!
    "AtlasFile": os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif"),
    # path to annotation file for structures
    "annotationfile":  os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annotation.tif"),
    "blendtype": "sigmoidal",  # False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
    # True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
    "intensitycorrection": False,
    "rawdata": True,  # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
    # Orientation: 1,2,3 means the same orientation as the reference and atlas files; #Flip axis with - sign (eg. (-1,2,3) flips x). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y); USE (3,2,1) for DVhorizotnal to sagittal. NOTE (TP): -3 seems to mess up the function and cannot seem to figure out why. do not use.
    "FinalOrientation": (3, 2, 1),
    "slurmjobfactor": 50,  # number of array iterations per arrayjob since max job array on SPOCK is 1000
}


def sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r, fEMP_size_r, fEMP_threshold_r,
                             fIP_method_r, fIP_size_r, dCSP_threshold_r,
                             tick, optimization_chunk=4, pth=False, rescale=False,
                             cleanup=True, **kwargs):
    """Function to sweep parameters

    final outputs will be saved in outputdirectory/parameter_sweep
    second copy will be saved in outputdirectory/parameter_sweep_jobid if cleanup=False

    Inputs:
        ----------------
        jobid: chunk of tissue to run (usually int between 20-30)
        #pth (optional): if pth to output folder after running package, function will load the param file automatically
        rescale (optional): str of dtype to rescale to. E.g.: "uint8"
        cleanup = T/F removes subfolders after
        optimization_chunk = this was the old "jobid" in this case it is the chunk of volume to look at
        kwargs (if not pth): "params" from run_clearmap_cluster.py
    """

    # make folder for final output:
    opt = kwargs["outputdirectory"]
    makedir(opt)
    out = opt+"/parameter_sweep"
    makedir(out)
    out0 = opt+"/parameter_sweep_jobid_{}".format(str(jobid).zfill(4))
    makedir(out0)

    rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold = [(rBP_size,
                                                                                             fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold) for rBP_size, fEMP_hmax,
                                                                                            fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r,
                                                                                                                                                                       fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r)][jobid]

    pth = out0+"/parametersweep_rBP_size{}_fEMP_hmax{}_fEMP_size{}_fEMP_threshold{}_fIP_method{}_fIP_size{}_dCSP_threshold{}.tif".format(rBP_size,
                                                                                                                                         fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold)

    if not os.path.exists(pth):
        try:
            # set params for sweep
            # Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
            kwargs["removeBackgroundParameter_size"] = (rBP_size, rBP_size)
            # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
            kwargs["findExtendedMaximaParameter_hmax"] = fEMP_hmax
            # size in pixels (x,y) for the structure element of the morphological opening
            kwargs["findExtendedMaximaParameter_size"] = fEMP_size
            # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
            kwargs["findExtendedMaximaParameter_threshold"] = fEMP_threshold
            # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
            kwargs["findIntensityParameter_method"] = fIP_method
            # (tuple)             size of the search box on which to perform the *method*
            kwargs["findIntensityParameter_size"] = (fIP_size, fIP_size, fIP_size)
            # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
            kwargs["detectCellShapeParameter_threshold"] = dCSP_threshold

            # tmp
            nkwargs = load_kwargs(kwargs["outputdirectory"])
            kwargs["outputdirectory"] = out0
            nkwargs.update(kwargs)
            pckloc = out0+"/param_dict.p"
            pckfl = open(pckloc, "wb")
            pickle.dump(nkwargs, pckfl)
            pckfl.close()

            # run cell detection
            sys.stdout.write(
                "\n\n\n           *****Iteration {} of {}*****\n\n\n".format(jobid, tick))
            sys.stdout.write("    Iteration parameters: {}     {}     {}     {}     {}     {}     {}".format(kwargs["removeBackgroundParameter_size"], kwargs["findExtendedMaximaParameter_hmax"], kwargs["findExtendedMaximaParameter_size"], kwargs[
                             "findExtendedMaximaParameter_threshold"],         kwargs["findIntensityParameter_method"],         kwargs["findIntensityParameter_size"],        kwargs["detectCellShapeParameter_threshold"]))
            celldetection_operations(optimization_chunk, testing=True, **kwargs)

            # list, load, and maxip
            raw = [xx for xx in listdirfull(out0+"/optimization/raw")
                   if "~" not in xx and ".db" not in xx]
            raw.sort()
            raw_im = np.squeeze(tifffile.imread(raw))
            raw_mx = np.max(raw_im, axis=0)
            bkg = [xx for xx in listdirfull(out0+"/optimization/background")
                   if "~" not in xx and "Thumbs.db" not in xx]
            bkg.sort()
            bkg_im = tifffile.imread(bkg)
            bkg_mx = np.max(bkg_im, axis=0)
            cell = [xx for xx in listdirfull(out0+"/optimization/cell")
                    if "~" not in xx and ".db" not in xx]
            cell.sort()
            cell_im = tifffile.imread(cell)
            cell_mx = np.max(cell_im, axis=0)

            # optional rescale:
            if rescale:
                raw_mx = rescale_intensity(raw_mx, in_range=str(raw_mx.dtype),
                                           out_range=rescale).astype(rescale)
                bkg_mx = rescale_intensity(bkg_mx, in_range=str(bkg_mx.dtype),
                                           out_range=rescale).astype(rescale)
                cell_mx = rescale_intensity(cell_mx, in_range=str(cell_mx.dtype),
                                            out_range=rescale).astype(rescale)

            # concatenate and save out:
            bigim = np.concatenate((raw_mx, bkg_mx, cell_mx), axis=1)
            del bkg, bkg_im, bkg_mx, cell, cell_im, cell_mx
            if cleanup:
                removedir(out0)
            if not cleanup:
                tifffile.imsave(pth, bigim, compress=1)

            #save in main
            npth = out+"/jobid_{}_rBPSize{}_fEMPHmax{}_fEMPSize{}_fEMPThreshold{}_fIPMethod{}_fIPSize{}_dCSPThreshold{}.tif".format(
                str(jobid).zfill(4), rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold)
            tifffile.imsave(npth, bigim.astype("uint16"), compress=1)

        except Exception as e:
            print("Error on: {}\n\nerror={}".format(pth, e))
            im = np.zeros((10, 10, 10))
            tifffile.imsave(pth, im, compress=1)
            with open(os.path.join(out, "errored_files.txt"), "a") as fl:
                fl.write("\n\n{}\n{}\n".format(pth, kwargs))
                fl.close

    return


# %%
if __name__ == "__main__":
    # parallelized for cluster
    print(sys.argv)
    stepid = int(sys.argv[1])

    # run step 1 to populate fullsizedata folder
    if stepid == 0:
        # make output folder
        makedir(params["outputdirectory"])
        # make parameter dictionary and pickle file:
        # e.g. single job assuming directory_determiner function has been properly set
        updateparams(os.getcwd(), **params)
        # copy folder into output for records
        if not os.path.exists(os.path.join(params["outputdirectory"], "ClearMapCluster")):
            shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "clearmap_cluster"), ignore=shutil.ignore_patterns(
                "^.git"))  # copy run folder into output to save run info
        # make planes
        for stepid in range(0, 30):
            arrayjob(stepid, cores=12, compression=1, **params)

    # run paramter sweep on full resolution data
    if stepid == 1:
        # get array ID
        print(os.environ["SLURM_ARRAY_TASK_ID"])
        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])  # int(sys.argv[2])
        ######################################################################################################
        # NOTE: To adjust parameter sweep, modify ranges below
        ######################################################################################################
        # evens seem to not be good  #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
        rBP_size_r = [7, 9, 11]
        # (float or None) h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
        fEMP_hmax_r = [None]
        # size in pixels (x,y) for the structure element of the morphological opening
        fEMP_size_r = [0]
        fEMP_threshold_r = [None]  # range(0,10)
        fIP_method_r = ["Max"]  # ["Max, "Mean"]
        fIP_size_r = [20]
        # [60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]#range(50, 200, 10)
        dCSP_threshold_r = [300, 500, 700]
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        # calculate number of iterations
        tick = 0
        for rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r,
                                                                                                            fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r):
            tick += 1
        sys.stdout.write("\n\nNumber of iterations is {}:".format(tick))

        # iterate through combination of parameters
        try:
            sweep_parameters_cluster(jobid, rBP_size_r, fEMP_hmax_r,
                                     fEMP_size_r, fEMP_threshold_r, fIP_method_r,
                                     fIP_size_r, dCSP_threshold_r, tick, optimization_chunk=20,
                                     cleanup=False, **params)
        except Exception as e:
            print("Jobid {}, Error given {}".format(jobid, e))

    # end server
    vdisplay.stop()
