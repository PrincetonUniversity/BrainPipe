#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:50:08 2016

@author: tpisano
"""

import os, sys, shutil
from xvfbwrapper import Xvfb; vdisplay = Xvfb(); vdisplay.start()
from tools.imageprocessing import preprocessing
from tools.registration.register import elastix_wrapper
from tools.utils.directorydeterminer import directorydeterminer

systemdirectory=directorydeterminer()

###set paths to data
###inputdictionary stucture: key=pathtodata value=list["xx", "##"] where xx=regch, injch, or cellch and ##=two digit channel number
#"regch" = channel to be used for registration, assumption is all other channels are signal
#"cellch" = channel(s) to apply cell detection
#"injch" = channels(s) to quantify injection site
#e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"], ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary={
os.path.join(systemdirectory, "LightSheetTransfer/tank/200110_pinto_sp4_20200106_1_3x_488_016na_1hfds_z5um_250msec_13-54-45"): 
    [["regch", "00"]]
}

####Required inputs
params={
"systemdirectory":  systemdirectory, #don"t need to touch
"inputdictionary": inputdictionary, #don"t need to touch
"outputdirectory": os.path.join(systemdirectory, "LightSheetData/tank-mouse/lucas/processed/sp4_20200106"),
"xyz_scale": (5, 5, 5), #(5.0,5.0,3), #micron/pixel: 5.0um/pix for 1.3x; 1.63um/pix for 4x
"tiling_overlap": 0.00, #percent overlap taken during tiling
"stitchingmethod": "blending", #"terastitcher", blending see below for details
"AtlasFile": os.path.join(systemdirectory, "LightSheetTransfer/atlas/allen_atlas/average_template_25_sagittal_forDVscans.tif"),
"annotationfile": os.path.join(systemdirectory, "LightSheetTransfer/atlas/allen_atlas/annotation_template_25_sagittal_forDVscans.tif"), ###path to annotation file for structures
"blendtype": "sigmoidal", #False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental
"intensitycorrection": True, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
"resizefactor": 3, ##in x and y #normally set to 5 for 4x objective, 3 for 1.3x obj
"rawdata": True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
"finalorientation":  ("2","1","0"), #Used to account for different orientation between brain and atlas. Assumes XYZ ("0","1","2) orientation. Pass strings NOT ints. "-0" = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation import fix_orientation; ("2","1","0") for horizontal to sagittal, Order of operations is reversing of axes BEFORE swapping axes.
"slurmjobfactor": 50 #number of array iterations per arrayjob since max job array on SPOCK is 1000
}

#####################################################################################################################################################
##################################################stitchingmethod####################################################################################
#####################################################################################################################################################
# "terastitcher": computationally determine overlap. See .py file and http://abria.github.io/TeraStitcher/ for details. NOTE THIS REQUIRES COMPILED SOFTWARE.
    #if testing terastitcher I strongly suggest adding to the parameter file
    #transfertype="copy", despite doubling data size this protects original data while testing
#"blending: using percent overlap to determine pixel overlap. Then merges using blendtype, intensitycorrection, blendfactor. This is not a smart algorithm
    
#####################################################################################################################################################
##################################################optional arguments for params######################################################################
#####################################################################################################################################################
# "regexpression":  r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", ###lavision preprocessed data
# "regexpression":  r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", lavision NONTILING + nonraw**
# "regexpression":  r"(.*)(.*C+)(.*)(.*Z+)(?P<z>[0-9]{1,4})(.*r+)(?P<ch>[0-9]{1,4})(.ome.tif)",
# "parameterfolder" : os.path.join(systemdirectory, "wang/pisano/Python/lightsheet/parameterfolder"), ##  * folder consisting of elastix parameter files with prefixes "Order<#>_" to specify application order
# "atlas_scale": (25, 25, 25), #micron/pixel, ABA is likely (25,25,25)
# "swapaxes" :  (0,2), #Used to account for different orientation between brain and atlas. 0=z, 1=y, 2=x. i.e. to go from horizontal scan to sagittal (0,2).
# "maskatlas": {"x": all, "y": "125:202", "z": "75:125"}; dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out. Occurs AFTER orientation change.
# "cropatlas": {"x": all, "y": "125:202", "z": "75:125"}; dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be REMOVED rather than zeroed out. THIS FUNCTION DOES NOT YET AFFECT THE ANNOTATION FILE
# "blendfactor" : 4, #only for sigmoidal blending, controls the level of sigmoidal; parameter that is passed to np"s linspace; defaults to 4. Higher numbers = steeper blending; lower number = more gradual blending
# "bitdepth": specify the fullsizedatafolder bitdepth output
# "secondary_registration" True (default) - register other channel(s) to registration channel (regch) then apply transform determined from regch->atlas
#                          useful if imaging conditions were different between channel and regch, i.e. added horizontal foci, sheet na...etc
#                          False - apply transform determined from regch->atlas to other channels. Use if channel of interest has very different pixel distribution relative regch (i.e. dense labeling)

#run scipt portions
if __name__ == "__main__":

    #get jobids from SLURM or argv
    print(sys.argv)
    stepid = int(sys.argv[1])
    if systemdirectory != "/home/wanglab/":
        print(os.environ["SLURM_ARRAY_TASK_ID"])
        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        jobid = int(sys.argv[2])


    #######################STEP 0 #######################
    #Make parameter dictionary and setup destination
    #####################################################
    if stepid == 0:
        ###make parameter dictionary and pickle file:
        preprocessing.generateparamdict(os.getcwd(), **params) # e.g. single job assuming directory_determiner function has been properly set
        #preprocessing.updateparams("/home/wanglab/wang/pisano/Python/lightsheet", svnm = "param_dict_local.p", **params) # make a local copy
        if not os.path.exists(os.path.join(params["outputdirectory"], "lightsheet")): 
            shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "lightsheet"), 
                            ignore=shutil.ignore_patterns(*(".pyc","CVS",".git","tmp",".svn", 
                                                            "TeraStitcher-Qt4-standalone-1.10.11-Linux"))) #copy run folder into output to save run info
        #os.system("rsync -av --exclude=".git/" ....)#
    #######################STEP 1 #######################
    #Stitch and preprocess each z plane
    #####################################################
    elif stepid == 1:
        if params["stitchingmethod"] not in ["terastitcher", "Terastitcher", "TeraStitcher"]:
            ###stitch based on percent overlap only ("dumb stitching"), and save files; showcelldetection=True: save out cells contours ovelaid on images
            preprocessing.arrayjob(jobid, cores=6, compression=1, **params) #process zslice numbers equal to slurmjobfactor*jobid thru (jobid+1)*slurmjobfactor
        else:
            #Stitch using Terastitcher "smart stitching"
            from tools.imageprocessing.stitch import terastitcher_from_params
            terastitcher_from_params(jobid=jobid, cores=6, **params)
            
    #######################STEP 2 #######################
    #Consolidate for Registration
    #####################################################
    elif stepid == 2:
        ###combine downsized ch and ch+cell files
        preprocessing.tiffcombiner(jobid, cores = 10, **params)
        
    #######################STEP 3 #######################
    #####################################################
    elif stepid == 3:
        elastix_wrapper(jobid, cores=12, **params) #run elastix

    vdisplay.stop()
