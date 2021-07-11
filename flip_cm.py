#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: ejdennis
"""

import os, sys, glob, shutil
import tifffile as tif 
import numpy as np


#listofdownsized = glob.glob('/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/*/imaging_request_1/rawdata/*/*downsized_for_atlas.tif')
listofdownsized = ["/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_E156/imaging_request_1/rawdata/resolution_3.6x/cell__downsized_for_atlas.tif",
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_E156/imaging_request_1/rawdata/resolution_3.6x/reg__downsized_for_atlas.tif",
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_M128/imaging_request_1/rawdata/resolution_3.6x/cell__downsized_for_atlas.tif",
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_M128/imaging_request_1/rawdata/resolution_3.6x/reg__downsized_for_atlas.tif"]


for filepath in listofdownsized:
	print(filepath)
	tif.imsave(filepath,np.fliplr(tif.imread(filepath)))
