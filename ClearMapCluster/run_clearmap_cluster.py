#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:50:08 2016

@author: tpisano
"""

import os, sys, shutil
from xvfbwrapper import Xvfb; vdisplay = Xvfb(); vdisplay.start()
from ClearMap.cluster.preprocessing import updateparams, arrayjob, process_planes_completion_checker
from ClearMap.cluster.directorydeterminer import directorydeterminer

systemdirectory=directorydeterminer()
###set paths to data
###inputdictionary stucture: key=pathtodata value=list["xx", "##"] where xx=regch, injch, or cellch and ##=two digit channel number
#"regch" = channel to be used for registration, assumption is all other channels are signal
#"cellch" = channel(s) to apply cell detection
#"injch" = channels(s) to quantify injection site
#"##" = when taking a multi channel scan following regexpression, the channel corresponding to the reg/cell/inj channel. I.e. name_of_scan_channel00_Z#### then use "00"
#e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"], ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary={
os.path.join(systemdirectory, "LightSheetTransfer/brody/z265"): [["regch", "00"]],
os.path.join(systemdirectory, "LightSheetTransfer/brody/z265"): [["cellch", "00"]]
}
####Required inputs

######################################################################################################
#NOTE: edit clearmap/parameter_file.py for cell some detection parameters, everything else is handled below
######################################################################################################

params={
"inputdictionary": inputdictionary, #don"t need to touch
"outputdirectory": os.path.join(systemdirectory, "LightSheetData/rat-brody/processed/201910_tracing/clearmap/z265"),
"resample" : False, #False/None, float(e.g: 0.4), amount to resize by: >1 means increase size, <1 means decrease
"xyz_scale": (1.63, 1.63, 10.0), #micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
"tiling_overlap": 0.00, #percent overlap taken during tiling
"AtlasFile" : os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif"), ###it is assumed that input image will be a horizontal scan with anterior being "up"; USE .TIF!!!!
"annotationfile" :  os.path.join(systemdirectory, "LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annotation.tif"), ###path to annotation file for structures
"AtlasResolution": (39,39,39), #um/voxel, optional resolution of atlas, used in resampling and will default to 25um if not provided
"blendtype" : "sigmoidal", #False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
"intensitycorrection" : False, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
"rawdata" : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
"FinalOrientation": (3, 2, 1), #Orientation: 1,2,3 means the same orientation as the reference and atlas files; #Flip axis with - sign (eg. (-1,2,3) flips x). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y); USE (3,2,1) for DVhorizotnal to sagittal. NOTE (TP): -3 seems to mess up the function and cannot seem to figure out why. do not use.
"slurmjobfactor": 50, #number of array iterations per arrayjob since max job array on SPOCK is 1000
"removeBackgroundParameter_size": (7,7), #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
"findExtendedMaximaParameter_hmax": None, # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
"findExtendedMaximaParameter_size": 0, # size in pixels (x,y) for the structure element of the morphological opening
"findExtendedMaximaParameter_threshold": None, # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
"findIntensityParameter_method": "Max", # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
"findIntensityParameter_size": (30,30,30), # (tuple)             size of the search box on which to perform the *method*
"detectCellShapeParameter_threshold": 500# (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
}
#####################################################################################################################################################
##################################################optional arguments for params######################################################################
#####################################################################################################################################################
#"regexpression":  r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", ###lavision preprocessed data
#"regexpression":  r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", lavision NONTILING**
#regexpression: "r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", lavision Channels and Z
#"AtlasResolution": (20,20,20), #um/voxel, optional resolution of atlas, used in resampling and will default to 25um if not provided
#"ResolutionAffineCFosAutoFluo": (16, 16, 16), #optional scaling for cfos to auto, will default to 16 isotropic
#"parameterfolder" : os.path.join(systemdirectory, "wang/pisano/Python/lightsheet/parameterfolder"), ##  * folder consisting of elastix parameter files with prefixes "Order<#>_" to specify application order
#"removeBackgroundParameter_size": (7,7), #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
#"findExtendedMaximaParameter_hmax": None, # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
#"findExtendedMaximaParameter_size": 5 # size in pixels (x,y) for the structure element of the morphological opening
#"findExtendedMaximaParameter_threshold": 0, # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
#"findIntensityParameter_method": "Max", # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
#"findIntensityParameter_size": (3,3,3), # (tuple)             size of the search box on which to perform the *method*
#"detectCellShapeParameter_threshold": 500 # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################


#run scipt portions
if __name__ == "__main__":

    #get job id from SLURM
    print(sys.argv)
    print(os.environ["SLURM_ARRAY_TASK_ID"])
    jobid = int(os.environ["SLURM_ARRAY_TASK_ID"]) #int(sys.argv[2])
    stepid = int(sys.argv[1])

    #######################STEP 0 #######################
    #####################################################
    if stepid == 0:
        ###make parameter dictionary and pickle file:
        updateparams(os.getcwd(), **params) # e.g. single job assuming directory_determiner function has been properly set
        #copy folder into output for records
        if not os.path.exists(os.path.join(params["outputdirectory"], "ClearMapCluster")): 
            shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "ClearMapCluster"), 
                            ignore=shutil.ignore_patterns("^.git")) #copy run folder into output to save run inf

    #######################STEP 1 #######################
    #####################################################
    elif stepid == 1:
        ###stitch, resample, and save files
        arrayjob(jobid, cores=5, compression=1, **params) #process zslice numbers equal to slurmjobfactor*jobid thru (jobid+1)*slurmjobfactor

    #######################STEP 2 #######################
    #####################################################

    elif stepid == 2:
        ###check to make sure all step 1 jobs completed properly
        if jobid == 0: process_planes_completion_checker(**params)
        #clearmap: load the parameters:
        from ClearMap.cluster.par_tools import resampling_operations
        resampling_operations(jobid, **params)

    #######################STEP 3 #######################
    #####################################################

    elif stepid == 3:
        #clearmap"s registration
        from ClearMap.cluster.par_tools import alignment_operations
        alignment_operations(jobid, **params)

    #######################STEP 4 #######################
    #####################################################
    #CELL DETECTION
    elif stepid == 4:
        #clearmap"s cell detection
        from ClearMap.cluster.par_tools import celldetection_operations
        celldetection_operations(jobid, **params)
    #######################STEP 5 #######################
    #####################################################
    #Consolidate Cell detection
    elif stepid == 5: 
        #clearmap:
        from ClearMap.cluster.par_tools import join_results_from_cluster
        join_results_from_cluster(**params)
    #######################STEP 6 #######################
    #####################################################
    #Finish Analysis
    elif stepid == 6:
        #clearmap analysis, for description of inputs check docstring ["output_analysis?"]:
        from ClearMap.cluster.par_tools import output_analysis
        output_analysis(threshold = (1500, 10000), row = (2,2), check_cell_detection = False, **params) #note: zmd has set threshold and 
        #row variable manually... see GDoc for more info?
