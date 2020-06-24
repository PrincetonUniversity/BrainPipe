#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:46:18 2018

@author: tpisano
"""

import numpy as np, sys, os
from skimage.external import tifffile
from skimage import filters
from skimage.morphology import binary_dilation, ball
from tools.registration.register import similarity_transform, transformix_command_line_call
from tools.utils.io import *


if __name__ == '__main__':
    
    #inputs
    dst = '/home/wanglab/wang/Jess/lightsheet_output/201810_cfos/dadult_pc_crusi_1/clearmap_cluster_output'
    atlas = '/home/wanglab/LightSheetTransfer/atlas/allen_atlas/average_template_25_sagittal_forDVscans.tif'
    auto = '/home/wanglab/wang/Jess/lightsheet_output/201810_cfos/dadult_pc_crusi_1/clearmap_cluster_output/autofluo_resampled.tif'
    ann = '/home/wanglab/LightSheetTransfer/atlas/allen_atlas/annotation_template_25_sagittal_forDVscans.tif'
    
    #run
    masked_atlas_pth, masked_ann_pth =mask_similarity_transformed_atlas(auto, ann, atlas, dst, verbose=True)

def mask_similarity_transformed_atlas(auto, ann, atlas, dst, verbose=True):
    '''
    Function to similarity transform atlas, mask based on ostu's method and save out
    
    Inputs
    -----------
    auto = path to autofluor image from clearmap cluster: './clearmap_cluster_output/autofluo_resampled.tif'
    atlas = path to atlas
    ann = path to annotation file
    dst = location to save files: './clearmap_cluster_output'
    
    Return
    ----------
    masked_atlas_pth
    masked_ann_pth
    
    '''
    
    #setup directory
    sim_dst = os.path.join(dst, 'similarity_transform'); makedir(dst)
    
    #do similarity transform on atlas to roughly align them - we might consider affine
    sim_atlas, tp = similarity_transform(fx = auto, mv = atlas, dst = sim_dst, nm = 'atlas_similarity_to_auto.tif', level='intermediate', cleanup=False)
    
    #transform ann
    transformix_command_line_call(src=ann, dst=sim_dst, transformfile = tp)
    sim_ann = os.path.join()
    
    #binarize auto
    if verbose:sys.stdout.write('Thresholding similarity tranformed atlas...'); sys.stdout.flush()
    mask = otsu_threshold(auto, verbose=verbose)

    #mask atlas
    if verbose: print('Masking...')
    masked_atlas = tifffile.imread(sim_atlas)
    masked_ann = tifffile.imread(sim_ann)
    masked_atlas[mask==0]=0
    masked_ann[mask==0]=0
    
    #save out
    masked_atlas_pth = os.path.join(dst, 'atlas_similarity_to_auto_masked.tif')
    masked_ann_pth = os.path.join(dst, 'ann_similarity_to_auto_masked.tif')
    if verbose: print('Saving as {}'.format(masked_atlas_pth))
    tifffile.imsave(masked_atlas_pth, masked_atlas)
    tifffile.imsave(masked_ann_pth, masked_ann)

    return masked_atlas_pth, masked_ann_pth

def otsu_threshold(src, otsu_factor = 0.4, gaus_selem=17, dilation_factor = 45, verbose=True):
    '''Function to threshold
    '''
    if verbose: sys.stdout.write('gaussian filtering...'); sys.stdout.flush()
    vol = filters.gaussian(tifffile.imread(src), gaus_selem)
    if verbose: sys.stdout.write('otsu thresholding...'); sys.stdout.flush()
    thresh=filters.threshold_otsu(vol) * float(otsu_factor)
    vol[vol<thresh] = 0
    vol[vol<0] = 1
    if verbose: sys.stdout.write('dilating...'); sys.stdout.flush()
    try:
        import cv2
        kernel = np.ones((dilation_factor,dilation_factor),np.uint8)
        vol = np.asarray([cv2.dilate(v,kernel,iterations = 1) for v in vol])
    except:
        vol = binary_dilation(vol.astype('bool'), ball(dilation_factor))
    return vol

