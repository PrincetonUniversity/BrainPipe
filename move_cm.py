#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: ejdennis
"""

import os, sys, glob, shutil

listofdownsized = glob.glob('/jukebox/LightSheetData/lightserv/pbibawi/pb_udisc*/*/imaging_request_1/rawdata/resolution_3.6x/*downsized_for_atlas.tif')

for filepath in listofdownsized:
    filepieces = filepath.split('/')
    shutil.copy(filepath,os.path.join('/scratch/ejdennis/cm2_brains/',"{}_{}".format(filepieces[6],filepieces[-1])))
