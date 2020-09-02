#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:50:08 2016

@author: tpisano
"""

from tools.utils.directorydeterminer import directorydeterminer
from tools.registration.register import elastix_wrapper
from tools.imageprocessing import preprocessing
import os
import sys
import shutil
from xvfbwrapper import Xvfb
vdisplay = Xvfb()
vdisplay.start()

systemdirectory = directorydeterminer()
# systemdirectory = "/home/emilyjanedennis"

# set paths to data
# inputdictionary stucture: key=pathtodata value=list["xx", "##"] where
# xx=regch, injch, or cellch and ##=two digit channel number
# "regch" = channel to be used for registration, assumption is all other
# channels are signal
# "cellch" = channel(s) to apply cell detection
# "injch" = channels(s) to quantify injection site
# e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"],
# ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary = {
    os.path.join(systemdirectory, "LightSheetData/lightserv/ejdennis/three_female_atlas_brains/three_female_atlas_brains-002/imaging_request_1/rawdata/200901_p002_1_1x_488_016na_1hfds_z10um_50msec_20povlp_13-47-38"):
    [["regch", "00"]]
    # ,
    # os.path.join(systemdirectory, "LightSheetTransfer/brody/z266"):
    # [["cellch", "00"]]
}

# Required inputs
params = {
    "systemdirectory":  systemdirectory,  # don"t need to touch
    "inputdictionary": inputdictionary,  # don"t need to touch
    "outputdirectory": os.path.join(systemdirectory, "scratch/ejdennis/f002"),
    # (5.0,5.0,3), #micron/pixel: 5.0um/pix for 1.3x; 1.63um/pix for 4x
    "xyz_scale": (5,5,10),
    "tiling_overlap": 0.20,  # percent overlap taken during tiling
    "stitchingmethod": "terastitcher",  # "terastitcher" or "blending"
    # "AtlasFile": os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif"),
    # path to annotation file for structures
    # "annotationfile": os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annotation.tif"),
    "AtlasFile": "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif",
    "annotationfile": "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annoatation.tif",
    "blendtype": "sigmoidal",  # False/None, "linear", or "sigmoidal"
    # blending between tiles, usually sigmoidal;
    # False or None for images where blending would be detrimental
    # True = calculate mean intensity of overlap between tiles shift higher
    # of two towards lower - useful for images where relative intensity
    # is not important (i.e. tracing=True, cFOS=False)
    "intensitycorrection": True,
    "resizefactor": 5,  # in x and y #normally set to 5 for 4x objective,
    # 3 for 1.3x obj
    "rawdata": True,  # set to true if raw data is taken from scope and
    # images need to be flattened; functionality for
    # rawdata =False has not been tested**
    # Used to account for different orientation between brain and atlas.
    # Assumes XYZ ("0","1","2) orientation.
    # Pass strings NOT ints. "-0" = reverse the order of the xaxis.
    # For better description see docstring from
    # tools.imageprocessing.orientation import fix_orientation;
    # ("2","1","0") for horizontal to sagittal,
    # Order of operations is reversing of axes BEFORE swapping axes.
    "finalorientation":  ("2", "1", "0"),
    "slurmjobfactor": 50,
    # number of array iterations per arrayjob
    # since max job array on SPOCK is 1000
    "transfertype": "copy"
}
print("outputdirectory")

# stitchingemthod
# "terastitcher": computationally determine overlap.
# See .py file and http://abria.github.io/TeraStitcher/ for details.
# NOTE THIS REQUIRES COMPILED SOFTWARE.
# if testing terastitcher I strongly suggest adding to the parameter file
# transfertype="copy", despite doubling data size this protects original
# data while testing
# "blending: using percent overlap to determine pixel overlap.
# Then merges using blendtype, intensitycorrection, blendfactor.
# This is not a smart algorithm

# additional optional params
# "parameterfolder" :
# "atlas_scale": (25, 25, 25), #micron/pixel, ABA is likely (25,25,25)
# "swapaxes" :  (0,2),
# "maskatlas": {"x": all, "y": "125:202", "z": "75:125"};
# "cropatlas": {"x": all, "y": "125:202", "z": "75:125"};
# "blendfactor" :
# "bitdepth":
# "secondary_registration"
# run scipt portions
if __name__ == "__main__":

    # get jobids from SLURM or argv
    print(sys.argv)
    stepid = int(sys.argv[1])
    if systemdirectory != "/home/emilyjanedennis/":
        print(os.environ["SLURM_ARRAY_TASK_ID"])
        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        jobid = int(sys.argv[2])

    # Make parameter dictionary and setup destination
    if stepid == 0:
        # make parameter dictionary and pickle file:
        # e.g. single job assuming directory_determiner function has
        # been properly set
        preprocessing.generateparamdict(os.getcwd(), **params)
        # preprocessing.updateparams("/", svnm = "param_dict_local.p",**params)
        # make a local copy
        if not os.path.exists(os.path.join(params["outputdirectory"],
                                           "lightsheet")):
            shutil.copytree(os.getcwd(), os.path.join(
                params["outputdirectory"],
                "lightsheet"),
                ignore=shutil.ignore_patterns(*(
                    ".pyc", "CVS",
                    ".git", "tmp", ".svn",
                    "TeraStitcher-Qt4-standalone-1.10.11-Linux")))
            # copy run folder into output to save run info

    # Stitch and preprocess each z plane
    elif stepid == 1:
        if params["stitchingmethod"] not in ["terastitcher"]:
            # stitch based on percent overlap only ("dumb stitching"),
            # and save files; showcelldetection=True:
            # save out cells contours ovelaid on images
            # process zslice numbers equal to
            # slurmjobfactor*jobid thru (jobid+1)*slurmjobfactor
            print("not terastitcher")
            preprocessing.arrayjob(jobid, cores=6, compression=1, **params)
        else:
            # Stitch using Terastitcher "smart stitching"
            from tools.imageprocessing.stitch import terastitcher_from_params
            terastitcher_from_params(jobid=jobid, cores=6, **params)
    # Consolidate for Registration
    elif stepid == 2:
        # combine downsized ch and ch+cell files
        preprocessing.tiffcombiner(jobid, cores=10, **params)

    elif stepid == 3:
        elastix_wrapper(jobid, cores=12, **params)  # run elastix

    vdisplay.stop()
