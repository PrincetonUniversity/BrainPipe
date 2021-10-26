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

from parameter_dictionary import params

#run scipt portions
if __name__ == "__main__":

    #get jobids from SLURM or argv
    print(sys.argv)
    stepid = int(sys.argv[1])
    # if systemdirectory != "/home/wanglab/":
    #     print(os.environ["SLURM_ARRAY_TASK_ID"])
    try:
        jobid = int(sys.argv[2])
    except:
        pass
    #     jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # else:


    #######################STEP 0 #######################
    #Make parameter dictionary and setup destination
    #####################################################
    if stepid == 0:
        ###make parameter dictionary and pickle file:
        preprocessing.generateparamdict(os.getcwd(), **params) # e.g. single job assuming directory_determiner function has been properly set
        if not os.path.exists(os.path.join(params["outputdirectory"], "lightsheet")): 
            shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "lightsheet"), 
                            ignore=shutil.ignore_patterns(*(".pyc","CVS",".git","tmp",".svn", 
                                                            "TeraStitcher-Qt4-standalone-1.10.11-Linux"))) #copy run folder into output to save run info
        
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
