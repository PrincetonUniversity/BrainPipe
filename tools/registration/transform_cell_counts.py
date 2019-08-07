#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:16:35 2018

@author: tpisano
"""
import os, sys
from tools.utils.io import makedir, load_dictionary, load_np, listdirfull
from tools.registration.transform_list_of_points import create_text_file_for_elastix, point_transformix, modify_transform_files, unpack_pnts
from tools.registration.register import change_transform_parameter_initial_transform, transformix_command_line_call
from tools.imageprocessing.orientation import fix_contour_orientation, fix_dimension_orientation
import numpy as np, pandas as pd
from skimage.external import tifffile

if __name__ == '__main__':
    #goal is to transform cooridnates, voxelize based on number of cells and overlay with reigstered cell signal channel...
    #inputs
    
    #3dunet cell dataframe
    dataframe = pd.read_csv('/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/3dunet_output/pooled_cell_measures/20170115_tp_bl6_lob6a_1000r_02_cell_measures.csv')
    
    #location to save out
    dst = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/3dunet_output'
    
    #EXAMPLE USING LIGHTSHEET - assumes marking centers in the 'raw' full sized cell channel. This will transform those centers into "atlas" space (in this case the moving image)
    #in this case the "inverse transform has the atlas as the moving image in the first step, and the autofluorescence channel as the moving image in the second step 
    transformfiles = [
            #'/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/elastix_inverse_transform/cellch_20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_resized_ch00_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.0.txt',
            #'/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/elastix_inverse_transform/cellch_20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_resized_ch00_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.1.txt'
            '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/elastix_inverse_transform/cellch_20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_resized_ch00_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.0.txt',
            '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/elastix_inverse_transform/cellch_20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_resized_ch00_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.1.txt',]
                      
    
    #NOTE - it seems that the registration of cell to auto is failing on occasion....thus get new files...################################
    ######################################
    
    lightsheet_parameter_dictionary = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/param_dict.p'
    
    verbose=True
    
    converted_points = generate_transformed_cellcount(dataframe, dst, transformfiles, lightsheet_parameter_dictionary, verbose=verbose)
    
    #load and convert to single voxel loc
    zyx = load_np(converted_points)
    zyx = np.asarray([str((int(xx[0]), int(xx[1]), int(xx[2]))) for xx in load_np(converted_points)])
    from collections import Counter
    zyx_cnt = Counter(zyx)
    
    #manually call transformix..
    c_rfe = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_resized_ch00_resampledforelastix.tif'
    transformed_dst = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/3dunet_output/transformed_points'
    transformfile = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/elastix/TransformParameters.1.txt'
    transformix_command_line_call(c_rfe, transformed_dst, transformfile)
    
    #cell_registered channel
    cell_reg = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/3dunet_output/transformed_points/result.tif')
    cell_cnn = np.zeros_like(cell_reg)
    for zyx,v in zyx_cnt.iteritems():
        z,y,x = [int(xx) for xx in zyx.replace('(','',).replace(')','').split(',')]
        cell_cnn[z,y,x] = v*100
    merged = np.stack([cell_cnn, cell_reg, np.zeros_like(cell_reg)], -1)
    tifffile.imsave('/home/wanglab/Downloads/merged.tif', merged)
    #out = np.concatenate([cell_cnn, cell_reg, ], 0)


#%%
def generate_transformed_cellcount(dataframe, dst, transformfiles, lightsheet_parameter_dictionary, verbose=False):
    '''Function to take a csv file and generate an input to transformix
    
    Inputs
    ----------------
    dataframe = preloaded pandas dataframe
    dst = destination to save files
    transformfiles = list of all elastix transform files used, and in order of the original transform****
    lightsheet_parameter_file = .p file generated from lightsheet package
    '''
    #set up locations
    transformed_dst = os.path.join(dst, 'transformed_points'); makedir(transformed_dst)
    
    #make zyx numpy arry
    zyx = dataframe[['z','y','x']].values
    
    #adjust for reorientation THEN rescaling, remember full size data needs dimension change releative to resample
    kwargs = load_dictionary(lightsheet_parameter_dictionary)
    vol = [xx for xx in kwargs['volumes'] if xx.ch_type =='cellch'][0]
    fullsizedimensions = get_fullsizedims_from_kwargs(kwargs) #don't get from kwargs['volumes'][0].fullsizedimensions it's bad! use this instead
    zyx = fix_contour_orientation(zyx, verbose=verbose, **kwargs) #now in orientation of resample
    zyx = points_resample(zyx, original_dims = fix_dimension_orientation(fullsizedimensions, **kwargs), resample_dims = tifffile.imread(vol.resampled_for_elastix_vol).shape, verbose = verbose)[:, :3]
   
    #make into transformix-friendly text file
    pretransform_text_file = create_text_file_for_elastix(zyx, transformed_dst)
        
    #copy over elastix files
    transformfiles = modify_transform_files(transformfiles, transformed_dst) 
    change_transform_parameter_initial_transform(transformfiles[0], 'NoInitialTransform')
   
    #run transformix on points
    points_file = point_transformix(pretransform_text_file, transformfiles[-1], transformed_dst)
    
    #convert registered points into structure counts
    converted_points = unpack_pnts(points_file, transformed_dst)   
    
    return converted_points

def points_resample(src, original_dims, resample_dims, verbose = False):
    '''Function to adjust points given resizing by generating a transform matrix
    
    ***Assumes ZYX and that any orientation changes have already been done.***
    
    src: numpy array or list of np arrays of dims nx3
    original_dims (tuple)
    resample_dims (tuple)
    '''
    src = np.asarray(src).astype('float')
    assert src.shape[-1] == 3, 'src must be a nx3 array'
            
    #apply scale diff
    for i in range(3):
        factor = (resample_dims[i]) / float(original_dims[i])
        src[:,i] = src[:,i] * float(factor)
        if verbose: sys.stdout.write('\nrescaling {} by {}'.format(['z','y','x'][i], factor))
    if verbose: sys.stdout.write('\nfirst three transformed pnts:\n{}\n'.format(src[0:3]))

    return src

def get_fullsizedims_from_kwargs(kwargs):
    '''fullsizedims of vols is incorrect when using terastitcher...this fixes that
    '''
    vol = [xx for xx in kwargs['volumes'] if xx.ch_type =='cellch'][0]
    zf = len(listdirfull(vol.full_sizedatafld_vol, '.tif'))
    yf,xf = tifffile.imread(listdirfull(vol.full_sizedatafld_vol, 'tif')[0]).shape
    return tuple((zf, yf, xf))
    
