#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:58:52 2019
@author: wanglab
"""

import os, sys, pandas as pd, numpy as np
from skimage.external import tifffile

def check_patchlist_length_equals_patches(**params):
    
    """ 
    checks to see whether the number of patches that are needed to be made
    are made in the array job
    """
    
    patchlength = len(params["patchlist"])
    input_chnks = os.path.join(params["data_dir"], "input_chnks")
    
    if len(os.listdir(input_chnks)) == patchlength: 
        sys.stdout.write("\ncorrect number of patches made!\n"); sys.stdout.flush()
    else:
        sys.stdout.write("\npatches made in step 1 are less than the patches needed. \
                         please check the array job range submitted and try again\n"); sys.stdout.flush()
