#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:26:38 2017

@author: tpisano
"""

from itertools import product
from ClearMap.cluster.par_tools import celldetection_operations
from ClearMap.cluster.preprocessing import makedir, listdirfull, removedir
from ClearMap.cluster.directorydeterminer import directorydeterminer
import tifffile, numpy as np, sys, os
from scipy.ndimage.interpolation import zoom
from skimage.exposure import rescale_intensity, equalize_hist
import matplotlib.pyplot as plt
import socket, cv2
#if socket.gethostname() == 'wanglab-cr8rc42-ubuntu': import hillshading
#%%
if __name__ == '__main__':

    #run param sweeps
    from ClearMap.cluster.parameter_sweep import sweep_parameters
    systemdirectory = directorydeterminer()
    params1={
    'systemdirectory': directorydeterminer(), #don't need to touch
    #'inputdictionary': inputdictionary, #don't need to touch
    'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha'),
    'resample' : False, #False/None, float(e.g: 0.4), amount to resize by: >1 means increase size, <1 means decrease
    'xyz_scale': (5.0, 5.0, 3), #micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
    'tiling_overlap': 0.00, #percent overlap taken during tiling
    'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!
    'annotationfile' :   os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/annotation_25_ccf2015_forDVscans.nrrd'), ###path to annotation file for structures
    'blendtype' : 'sigmoidal', #False/None, 'linear', or 'sigmoidal' blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
    'intensitycorrection' : False, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
    'rawdata' : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
    'FinalOrientation': (3, 2, 1), #Orientation: 1,2,3 means the same orientation as the reference and atlas files; #Flip axis with - sign (eg. (-1,2,3) flips x). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y); USE (3,2,1) for DVhorizotnal to sagittal. NOTE (TP): -3 seems to mess up the function and cannot seem to figure out why. do not use.
    'slurmjobfactor': 3, #number of array iterations per arrayjob since max job array on SPOCK is 1000
    'removeBackgroundParameter_size': (7,7), #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
    'findExtendedMaximaParameter_hmax': None, # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
    'findExtendedMaximaParameter_size': 5, # size in pixels (x,y) for the structure element of the morphological opening
    'findExtendedMaximaParameter_threshold': 0, # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
    'findIntensityParameter_method': 'Max', # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
    'findIntensityParameter_size': (3,3,3), # (tuple)             size of the search box on which to perform the *method*
    'detectCellShapeParameter_threshold': 500 # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
    }

    params2={
    'systemdirectory': directorydeterminer(), #don't need to touch
    #'inputdictionary': inputdictionary, #don't need to touch
    'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10'),
    'resample' : False, #False/None, float(e.g: 0.4), amount to resize by: >1 means increase size, <1 means decrease
    'xyz_scale': (5.0, 5.0, 3), #micron/pixel; 1.3xobjective w/ 1xzoom 5um/pixel; 4x objective = 1.63um/pixel
    'tiling_overlap': 0.00, #percent overlap taken during tiling
    'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!
    'annotationfile' :   os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/annotation_25_ccf2015_forDVscans.nrrd'), ###path to annotation file for structures
    'blendtype' : 'sigmoidal', #False/None, 'linear', or 'sigmoidal' blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental;
    'intensitycorrection' : False, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
    'rawdata' : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
    'FinalOrientation': (3, 2, 1), #Orientation: 1,2,3 means the same orientation as the reference and atlas files; #Flip axis with - sign (eg. (-1,2,3) flips x). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y); USE (3,2,1) for DVhorizotnal to sagittal. NOTE (TP): -3 seems to mess up the function and cannot seem to figure out why. do not use.
    'slurmjobfactor': 3, #number of array iterations per arrayjob since max job array on SPOCK is 1000
    'removeBackgroundParameter_size': (7,7), #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
    'findExtendedMaximaParameter_hmax': None, # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
    'findExtendedMaximaParameter_size': 5, # size in pixels (x,y) for the structure element of the morphological opening
    'findExtendedMaximaParameter_threshold': 0, # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
    'findIntensityParameter_method': 'Max', # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
    'findIntensityParameter_size': (3,3,3), # (tuple)             size of the search box on which to perform the *method*
    'detectCellShapeParameter_threshold': 500 # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
    }

    #run
    from ClearMap.cluster.parameter_sweep import sweep_parameters
    for params in [params1, params2]:
        sweep_parameters(29, **params)

    #combine results
    from ClearMap.cluster.parameter_sweep import get_parameter_sweep_results
    lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10', '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha']
    dst =  '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_ctl'
    get_parameter_sweep_results(lst, dst, resize = 1, hillshade=True, plot=False, concatenate=True)

    #output single image
    from ClearMap.cluster.parameter_sweep import crop_to_dims
    im_lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/raw_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold50_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold200_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size7_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold50_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size9_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold110_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size13_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold150_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size15_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold150_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size17_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold150_hillshading.png']

    im = crop_to_dims(im_lst, xdim='965:1790', ydim='220:2160')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/background_concatenated_3-17.png'
    cv2.imwrite(sv,im)

    im_lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/raw_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold200_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold60_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold75_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold90_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold105_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold115_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold130_hillshading.png']
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold145_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold160_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold175_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold190_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold205_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold220_hillshading.png']
    im = crop_to_dims(im_lst, xdim='965:1790', ydim='220:2160')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/bkgd5_cells_concatenated_60-130.png'
    cv2.imwrite(sv,im)

    im_lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/raw_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_ims_background/bkgrdsubtract_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold60_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold75_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold90_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold105_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold115_hillshading.png',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold130_hillshading.png']
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold145_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold160_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold175_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold190_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold205_hillshading.png',
              #'/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/param_sweep_png_cells/cells_parametersweep_rBP_size11_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold220_hillshading.png']
    im = crop_to_dims(im_lst, xdim='965:1790', ydim='220:2160')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/bkgd11_cells_concatenated_60-130.png'
    cv2.imwrite(sv,im)
#%%
    #round 2:
    #ctl
    im_lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold50.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size7_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold150.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size9_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold110.tif']
    im = crop_to_dims([im_lst[0]], xdim='0:2160', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_raw'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2160:4320', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_bkgd'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2440:3250', ydim='475:900', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_bkgd_zoom'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2640:3050', ydim='475:775', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_bkgd_zoomx2'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='480:890', ydim='475:775', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_raw_zoomx2'
    tifffile.imsave(sv+'.tif',im)
    #exp
    im_lst = ['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold50.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10/optimization/parameter_sweep/parametersweep_rBP_size5_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10/optimization/parameter_sweep/parametersweep_rBP_size7_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold150.tif',
              '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_mk10/optimization/parameter_sweep/parametersweep_rBP_size9_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold110.tif']
    im = crop_to_dims([im_lst[0]], xdim='0:2160', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/exp_raw'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2160:4320', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/exp_bkgd'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2440:3250', ydim='475:900', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/exp_bkgd_zoom'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='2640:3050', ydim='475:775', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/exp_bkgd_zoomx2'
    tifffile.imsave(sv+'.tif',im)

    #threshold
    im_lst=['/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold80.tif',
            '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold100.tif',
            '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold120.tif',
            '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_par_201701_tpalpha/optimization/parameter_sweep/parametersweep_rBP_size3_fEMP_hmaxNone_fEMP_size5_fEMP_thresholdNone_fIP_methodMax_fIP_size5_dCSP_threshold140.tif']
    im = crop_to_dims(im_lst, xdim='4320:', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_cells'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='4600:5410', ydim='475:900', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_cells_zoom'
    tifffile.imsave(sv+'.tif',im)
    im = crop_to_dims(im_lst, xdim='4800:5210', ydim='475:775', im_type='tif')
    sv = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/presentation_images/ctl_cells_zoomx2'
    tifffile.imsave(sv+'.tif',im)

#%%

def sweep_parameters(jobid, pth=False, rescale=False, **kwargs):
    '''Function to sweep parameters


    Inputs:
        ----------------
        jobid: chunk of tissue to run (usually int between 20-30)
        #pth (optional): if pth to output folder after running package, function will load the param file automatically
        rescale (optional): str of dtype to rescale to. E.g.: 'uint8'
        kwargs (if not pth): 'params' from run_clearmap_cluster.py
    '''

    #if pth: sys.path.append(pth+'/clearmap_cluster'); from run_clearmap_cluster import params; kwargs = params

    #set param sweeps
    rBP_size_r = range(3,11,2) #[5, 11] #range(5,19,2) ###evens seem to not be good
    fEMP_hmax_r = [None]#[None, 5, 10, 20, 40]
    fEMP_size_r = [5]#range(3,8)
    fEMP_threshold_r = [None] #range(0,10)
    fIP_method_r = ['Max'] #['Max, 'Mean']
    fIP_size_r = [5]#range(1,5)
    dCSP_threshold_r = range(50,230,15)#[60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]#range(50, 200, 10)

    # calculate number of iterations
    tick = 0
    for rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r, fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r):
        tick +=1

    sys.stdout.write('\n\nNumber of iterations is {}:'.format(tick))

    #make folder for final output:
    opt = kwargs['outputdirectory']+'/optimization'; out = opt+'/parameter_sweep'; makedir(out)
    makedir(opt)#; removedir(out); makedir(out)

    ntick = 0
    for rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold in product(rBP_size_r, fEMP_hmax_r, fEMP_size_r, fEMP_threshold_r, fIP_method_r, fIP_size_r, dCSP_threshold_r):

        pth = out+'/parametersweep_rBP_size{}_fEMP_hmax{}_fEMP_size{}_fEMP_threshold{}_fIP_method{}_fIP_size{}_dCSP_threshold{}.tif'.format(rBP_size, fEMP_hmax, fEMP_size, fEMP_threshold, fIP_method, fIP_size, dCSP_threshold)

        if not os.path.exists(pth):

            try:

                #set params for sweep
                kwargs['removeBackgroundParameter_size'] = (rBP_size, rBP_size) #Remove the background with morphological opening (optimised for spherical objects), e.g. (7,7)
                kwargs['findExtendedMaximaParameter_hmax'] = fEMP_hmax # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
                kwargs['findExtendedMaximaParameter_size'] = fEMP_size # size in pixels (x,y) for the structure element of the morphological opening
                kwargs['findExtendedMaximaParameter_threshold'] = fEMP_threshold # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
                kwargs['findIntensityParameter_method'] =  fIP_method # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
                kwargs['findIntensityParameter_size'] = (fIP_size,fIP_size,fIP_size) # (tuple)             size of the search box on which to perform the *method*
                kwargs['detectCellShapeParameter_threshold'] = dCSP_threshold # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated

                #run cell detection
                ntick+=1
                sys.stdout.write('\n\n\n           *****Iteration {} of {}*****\n\n\n'.format(ntick, tick))
                sys.stdout.write('    Iteration parameters: {}     {}     {}     {}     {}     {}     {}'.format(kwargs['removeBackgroundParameter_size'], kwargs['findExtendedMaximaParameter_hmax'], kwargs['findExtendedMaximaParameter_size'], kwargs['findExtendedMaximaParameter_threshold'],         kwargs['findIntensityParameter_method'],         kwargs['findIntensityParameter_size'],        kwargs['detectCellShapeParameter_threshold']))
                celldetection_operations(jobid, testing = True, **kwargs)

                #list, load, and maxip
                if ntick == 1: raw = [xx for xx in listdirfull(opt+'/raw') if '~' not in xx and '.db' not in xx]; raw.sort(); raw_im = np.squeeze(tifffile.imread(raw)); raw_mx = np.max(raw_im, axis = 0)
                bkg = [xx for xx in listdirfull(opt+'/background') if '~' not in xx and 'Thumbs.db' not in xx]; bkg.sort(); bkg_im = tifffile.imread(bkg); bkg_mx = np.max(bkg_im, axis = 0)
                cell = [xx for xx in listdirfull(opt+'/cell') if '~' not in xx and '.db' not in xx]; cell.sort(); cell_im = tifffile.imread(cell); cell_mx = np.max(cell_im, axis = 0)

                #optional rescale:
                if rescale:
                    raw_mx = rescale_intensity(raw_mx, in_range=str(raw_mx.dtype), out_range=rescale).astype(rescale)
                    bkg_mx = rescale_intensity(bkg_mx, in_range=str(bkg_mx.dtype), out_range=rescale).astype(rescale)
                    cell_mx = rescale_intensity(cell_mx, in_range=str(cell_mx.dtype), out_range=rescale).astype(rescale)


                #concatenate and save out:
                bigim = np.concatenate((raw_mx, bkg_mx, cell_mx), axis = 1); del bkg, bkg_im, bkg_mx, cell, cell_im,cell_mx
                tifffile.imsave(pth, bigim, compress = 1)

            except Exception, e:
                print ('Error on: {}, error: {}'.format(pth,e))
                im = np.zeros((10,10,10))
                tifffile.imsave(pth, im, compress = 1)
                with open(os.path.join(out, 'errored_files.txt'), 'a') as fl:
                    fl.write('\n\n{}\n{}\n'.format(pth, kwargs))
                    fl.close



    return

#%%
def get_parameter_sweep_results(args, dst, resize=False, plot = False, hillshade=False, concatenate=False):
    '''

    If passing several parameter sweeps results from different brains this assumes that each had the same sweep parameters.

    Inputs
    ----------
    args = list of folders that are output of sweep_parameters function
    dst = pth to save folder
    resize = (optional; int) factor to resample by. smaller<1<larger
    plot (optional): save matplotlib figure
    hillshade (optional), if true apply hillshading
    concatenate (optional), if true, concatenate all into a single tifffile

    Outputs
    ----------
    combined tiff of images
    '''
    removedir(dst); makedir(dst)
    #get list of sublist of fls
    lst=[]
    for pth in args:
        fls = listdirfull(pth+'/optimization/parameter_sweep'); fls.sort()
        lst.append(fls)

    if not resize: resize = 1
    if tiff_stacks:
        tiff_stacks = dst+'/tiff_stacks'; makedir(tiff_stacks)

    #load and concatenate files:
    sys.stdout.write('Loading files....'); sys.stdout.flush()
    for xx in range(len(lst[0])):
        for yy in range(len(lst)):
            im = zoom(tifffile.imread(lst[yy][xx]), resize)
            if tiff_stacks: tifffile.imsave(tiff_stacks+lst[yy][xx][lst[yy][xx].rfind('/'):], im)
            if yy == 0: imm = im
            if yy > 0: imm = np.concatenate((imm, im), axis = 0)
        if hillshading:
            if xx==0:
                plt.ioff()
                xdim = imm.shape[1]
                hillshading.imshow_hs(imm[:,0:xdim/3], save = dst+'/'+'raw_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4]+'_hillshading'); plt.close()
            hillshading.imshow_hs(imm[:,xdim/3:2*xdim/3], save = dst+'/'+'bkgrdsubtract_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4]+'_hillshading'); plt.close()
            hillshading.imshow_hs(imm[:,2*xdim/3:3*xdim/3], save = dst+'/'+'cells_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4]+'_hillshading'); plt.close()

        if plot:
            if xx==0:
                plt.ioff()
                xdim = imm.shape[1]
                plt.imshow(imm[:,0:xdim/3]); plt.savefig(dst+'/'+'raw_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4], dpi=200); plt.close()
            plt.imshow(imm[:,xdim/3:2*xdim/3]); plt.savefig(dst+'/'+'bkgrdsubtract_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4], dpi=200); plt.close()
            plt.imshow(imm[:,2*xdim/3:3*xdim/3]); plt.savefig( dst+'/'+'cells_'+lst[yy][xx][lst[yy][xx].rfind('/')+1:-4], dpi=200); plt.close()

        if concatenate:
            if xx == 0:
               bigim = np.zeros((len(lst[0]), imm.shape[0], imm.shape[1]))
               bigim[0,...] = imm
            if xx > 0: bigim[xx,...] = imm
        if xx % 10 == 0: sys.stdout.write('\n{} of {}'.format(xx, len(lst[0]))); sys.stdout.flush()


    #fix hillshade so that you can save out image. AND FIX SCALING (RESIZE OUTS 32 BIT)

    if not hillshade:
        tifffile.imsave(dst+'/parameter_sweep_output.tif', bigim.astype('uint16'))
        sys.stdout.write('Saved as {}'.format(dst+'/parameter_sweep_output.tif')); sys.stdout.flush()


    return

#%%

def crop_to_dims(im_lst, xdim=':', ydim=':', im_type='png'):
    '''Function to crop results and generate an image. Concatenates in xdimension

    Inputs
    --------------
    im_lst: list of .pngs to concatenate
    xdim(str): pixels to keep, e.g. '75:125'
    ydim(str): pixels to keep, e.g. '75:125'
    im_type: 'png', 'tif' or 'tiff'

    Returns
    ------------
    numpy array of image
    '''

    import cv2, tifffile
    lst=[]
    if im_type == 'png':
        for xx in im_lst:
            #load im
            exec('im = cv2.imread(xx, 1)[{},{}]'.format(ydim, xdim)) #YXC

            #concatenate
            try:
                bigim = np.concatenate((bigim, im), axis = 1)
            except:
                bigim = im
    elif im_type == 'tif' or im_type == 'tiff':
        for xx in im_lst:
            #load im
            exec('im = tifffile.imread(xx)[{},{}]'.format(ydim, xdim)) #YXC

            #concatenate
            try:
                bigim = np.concatenate((bigim, im), axis = 1)
            except:
                bigim = im

    return bigim
