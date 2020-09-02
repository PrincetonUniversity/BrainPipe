#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:10:18 2019

@author: wanglab
"""

import os, tifffile, cv2, numpy as np, multiprocessing as mp, sys, shutil
from scipy.ndimage import zoom
sys.path.append("/jukebox/wang/zahra/python/BrainPipe")
from tools.utils.io import load_kwargs
from tools.imageprocessing.orientation import fix_orientation

def downsize(pln, dst, final_res = 25, X_fullres_dim = 2160, Y_fullres_dim = 4018):
    """
    function to take full-res volume of a whole brain image and downsize to a specific resolution
    NOTE: this only does the downsizing in XY, see 'export_to_tiff' for the final downsizing in Z
    to make an isotropic volumes
    run using parallel processing
    params:
        pln = full-res z-plane of the brain being downsized
        dst = directory to which you are saving downsized planes
        final_res = the resolution in microns you want the atlas to be in
        X_fullres_dim = the dimensions of images in X
        Y_fullres_dim = the dimensions of images in Y
    """

    print(os.path.basename(pln))
    img = tifffile.imread(pln)
    dims = (int(np.ceil(X_fullres_dim/(final_res/6.5/1.1))), int(np.ceil(Y_fullres_dim/(final_res/6.5/1.1)))) #dims of most of the ones i imaged
    #note, not all brains will have these dimensions in XY due to 3D stitiching, but they should be farily close so this
    #approximation is probably ok for now
    img_resz = cv2.resize(img, dims)
    tifffile.imsave(os.path.join(dst, os.path.basename(pln)), img_resz)

def run_downsizing(pth, cores = 12):
    """
    function to take a whole brain volume processed using BrainPipe and downsize it to a
    specific resolution in XY
    useful if you are trying to make an atlas with a specific final resolution (e.g. 25um in XYZ)
    """
    print(os.path.basename(pth))
    kwargs = load_kwargs(pth) #note: this function specifically relies on the paramter dictionary made when processing
    #if you move the processed directory to another location, the paths initially saved in this dictionary will be incorrect
    #so suggest you do not move processed directories, or see the BrainPipe scripts on how to correct it if you did
    regvol = [xx for xx in kwargs["volumes"] if xx.ch_type == "regch"][0]
    fszdt = regvol.full_sizedatafld_vol
    dst = os.path.join(pth, "downsized_for_atlas") #set destination for downsized planes
    if not os.path.exists(dst): os.mkdir(dst) #make dest directory
    plns = [os.path.join(fszdt, xx) for xx in os.listdir(fszdt) if "tif" in xx]; plns.sort()
    iterlst = [(pln, dst) for pln in plns]
    p = mp.Pool(cores)
    p.starmap(downsize, iterlst)

def export_to_tiff(pth, dst, z_step = 10, final_res = 25):
    """
    function to take downsized 2D planes and make them into a 3D isotropic volumes
    AKA downsize in Z and export to tif
    params:
        pth = lightsheet processed directory
        dst = secondary location to save iso volumes for registration
        z_step = z-steps used during imaging
        final_res = resolution of isotropic volume you want your atlas to be
    """
    fld = os.path.join(pth, "downsized_for_atlas")
    print(os.path.basename(pth))
    plns = [os.path.join(fld, xx) for xx in os.listdir(fld)]; plns.sort()
    arr = np.zeros((len(plns), tifffile.imread(plns[0]).shape[0], tifffile.imread(plns[0]).shape[1]))
    for i,pln in enumerate(plns):
        arr[i] = tifffile.imread(pln)
        if i%100 == 0: print(i)
    tifffile.imsave(os.path.join(pth, "downsized_for_atlas.tif"), arr.astype("uint16"))
    #remove folder with indiviudal tifs
    shutil.rmtree(fld)

    #make iso volume
    factor = z_step/final_res #e.g. downsizing z by 10um (z step) / 25 um (desired resolution)
    arr_dwnsz = zoom(arr, (factor, 1, 1), order = 1) #horizontal image
    arr_dwnsz_sag = fix_orientation(arr_dwnsz, axes = ("2", "1", "0"))
    tifffile.imsave(os.path.join(dst, os.path.basename(pth)+".tif"), arr_dwnsz_sag.astype("uint16"))

    return

if __name__ == "__main__":

    #these parameters are for running an the array job on the cluster
    #e.g. sbatch --array=0-50 mk_ls_atl.sh
    #to run locally, turn the variable 'jobid' into a for loop and loop through brains one-by-one
    print(os.environ["SLURM_ARRAY_TASK_ID"])
    jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])

    src = "/jukebox/LightSheetData/brodyatlas/processed"
    dst = "/jukebox/LightSheetData/brodyatlas/atlas/2019_meta_atlas/volumes"

    brains = ["k293"]
#            ["a235",
#             "a237",
#             "c223",
#             "c514",
#             "c515",
#             "c516",
#             "e106",
#             "f119",
#             "h170",
#             "h208",
#             "k281",
#             "k292",
#             "k301",
#             "k302",
#             "k303",
#             "k304",
#             "k307",
#             "w118",
#             "w128"]

    pths = [os.path.join(src, xx) for xx in brains]

    pth = pths[jobid]
    print(pth)
    #run
    run_downsizing(pth)
    #FIXME: can combine these 2 functions
    export_to_tiff(pth, dst)
