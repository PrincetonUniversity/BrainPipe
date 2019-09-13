#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:05:41 2019

@author: wanglab
"""

path_to_lighsheet_folder = '/home/kellyms/wang/seagravesk/toms_lightsheet_software_20190104/lightsheet'


import os, sys, shutil, numpy as np, pandas as pd
import datetime
from collections import Counter
sys.path.append(path_to_lighsheet_folder)
from tools.registration.register import transformix_command_line_call, transformed_pnts_to_allen_helper_func, count_structure_lister
from tools.utils.io import makedir, load_np, listall, load_kwargs
from skimage.external import tifffile
from tools.registration.transform_cell_counts import generate_transformed_cellcount, get_fullsizedims_from_kwargs, points_resample
from tools.registration.transform_list_of_points import modify_transform_files
from tools.imageprocessing.orientation import fix_contour_orientation, fix_dimension_orientation

def load_cell_centers(src):
        ###LOAD INTO DATAFRAME
    if src.endswith('.csv'):
        dataframe = pd.read_csv(src) #3dunet cell dataframe
        zyx = dataframe[['z','y','x']].values #make zyx numpy arry
    elif src.endswith('.mat'):
        from scipy.io import loadmat
        zyx = loadmat(src)['cell_centers_orig_coord'] #Kelly's .mat files
        dataframe = pd.DataFrame(data=zyx, columns = ['z','y','x'])
    else:
        zyx = np.load(src)
        dataframe = pd.DataFrame(data=zyx, columns = ['z','y','x'])
    return zyx, dataframe

if __name__ == '__main__':
    verbose = True
    cleanup = True #remove all the elastix internal stuff that might not be needed
    
    #path to csv file containing the columns 'fld', 'prefix_path_to_mat_file', 'dst'
    path_to_csv_file = '/home/kellyms/wang/seagravesk/lightsheet/path_file_for_toms_processing_code_20190909.csv'
        
    #file name suffix
    mat_file_suffix = 'sliding_diff_peak_find_99percentile_20190227_format2.mat'
    
    #destination suffix
    dst_suffix = 'cell_region_assignment_99percentile_no_erosion_20190909'
    
    #annotation file
    ann_pth = '/home/kellyms/LightSheetTransfer/atlas/allen_atlas/annotation_template_25_sagittal_forDVscans.tif'
    #ann_pth = '/home/kellyms/wang/seagravesk/lightsheet/annotation_files/annotation_template_25_sagittal_forDVscans_50umEdge_50umVentricle_erosion.tif'
    id_table = '/home/kellyms/LightSheetTransfer/atlas/allen_atlas/allen_id_table.xlsx'
   
    #type of registration to be done when tranforming cell counts from signal into atlas space "the goal of this script"
    generate_downsized_overlay = True #if you'd like to run this part
    # cell_transform_type = 'all', affine and reg for both auto+atlas, and signal+auto
    # cell_transform_type = 'single': don't consider reg between sig+auto at all. Only auto+atlas
    cell_transform_type = 'affine_only_reg_to_sig';#both for reg+atlas, and only affine for sig and reg. This is what clearmap does
    
    #QC - overlaying single *volume* transformed into atlas space WITH cell *centers* point transformed into atlas space.
    generate_registered_overlay = True #if you'd like to run this part
    #this isn't a perfect Quality control and you can expect some pixel shifts
    #qc_overlay_transform_type = 'affine_only_reg_to_sig';#both for reg+atlas, and only affine for sig and reg. This is what clearmap does
    #qc_overlay_transform_type = 'all' #both for auto+atlas and for sig+auto
    qc_overlay_transform_type = 'single'; #don't consider reg with sig at all

    
    #%%
    ###############################
    #run, don't touch below
    ###############################
    df = pd.read_csv(path_to_csv_file)    
    for i,row in df.iterrows():
        
        #set up destination
        dst = os.path.join(row['dst'], dst_suffix); makedir(dst)
        
        #lightsheet package output folder
        fld = row['fld']
        
        #path to mat file
        cell_centers_path = os.path.join(row['prefix_path_to_mat_file'], mat_file_suffix)
        
        print('\n\n_________________\nFolder: {}\nDestination: {}\nMatfile: {}\n'.format(fld, dst, cell_centers_path))

        #loads
        kwargs = load_kwargs(fld)
        regvol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0]
        cellvol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch'][0]
        dst0 = os.path.join(dst, os.path.basename(fld)); makedir(dst0)
        dst1 = os.path.join(dst0, 'elastix'); makedir(dst1)
        
        #####generate a downsized version######
        if generate_downsized_overlay:
            #load centers
            zyx, dataframe = load_cell_centers(cell_centers_path)
            
            cellvolloaded = tifffile.imread(cellvol.resampled_for_elastix_vol)
            cnn_cellvolloaded = np.zeros_like(cellvolloaded)
            #adjust for reorientation THEN rescaling, remember full size data needs dimension change releative to resample
            fullsizedimensions = get_fullsizedims_from_kwargs(kwargs) #don't get from kwargs['volumes'][0].fullsizedimensions it's bad! use this instead
            zyx = fix_contour_orientation(zyx, verbose=verbose, **kwargs) #now in orientation of resample
            zyx = points_resample(zyx, original_dims = fix_dimension_orientation(fullsizedimensions, **kwargs), 
                                  resample_dims = tifffile.imread(cellvol.resampled_for_elastix_vol).shape, verbose = verbose)[:, :3]
            zyx = np.asarray([str((int(xx[0]), int(xx[1]), int(xx[2]))) for xx in zyx])
            zyx_cnt = Counter(zyx)
            #now overlay
            for zyx,v in zyx_cnt.items():
                z,y,x = [int(xx) for xx in zyx.replace('(','',).replace(')','').split(',')]
                try:
                    cnn_cellvolloaded[z,y,x] = v*100
                except Exception as e:
                    print(e)
            merged = np.stack([cnn_cellvolloaded, cellvolloaded, np.zeros_like(cellvolloaded)], -1)
            merged = np.swapaxes(merged, 0,2)#reorient to horizontal
            tifffile.imsave(os.path.join(dst, 'generate_downsized_overlay_{}_points_merged_resampled_for_elastix.tif'.format(os.path.basename(fld))), merged)         
        
        #EXAMPLE USING LIGHTSHEET - assumes marking centers in the 'raw' full sized cell channel. This will transform those centers into "atlas" space (in this case the moving image)
        #in this case the "inverse transform has the atlas as the moving image in the first step, and the autofluorescence channel as the moving image in the second step 
        r2s0 = [xx for xx in listall(cellvol.inverse_elastixfld, 'reg2sig_TransformParameters.0.txt') if 'cellch' in xx][0]
        r2s1 = [xx for xx in listall(cellvol.inverse_elastixfld, 'reg2sig_TransformParameters.1.txt') if 'cellch' in xx][0]
        a2r0 = [xx for xx in listall(cellvol.inverse_elastixfld, 'atlas2reg2sig/atlas2reg_TransformParameters.0.txt') if 'cellch' in xx][0]
        a2r1 = [xx for xx in listall(cellvol.inverse_elastixfld, 'atlas2reg2sig/atlas2reg_TransformParameters.1.txt') if 'cellch' in xx][0]
        if cell_transform_type == 'all':
            transformfiles = [r2s0, r2s1, a2r0, a2r1]
        elif cell_transform_type == 'single':
            transformfiles = [a2r0, a2r1]
        elif cell_transform_type == 'affine_only_reg_to_sig':    
            transformfiles = [r2s0, a2r0, a2r1]
        transformfiles = modify_transform_files(transformfiles, dst = dst1)
           
        #load centers and convert points
        zyx, dataframe = load_cell_centers(cell_centers_path)
        converted_points = generate_transformed_cellcount(dataframe, dst1, transformfiles, lightsheet_parameter_dictionary=os.path.join(fld, 'param_dict.p'), verbose=verbose)
        shutil.copy(converted_points, os.path.join(dst))
        
        #align to annotation
        point_lst = transformed_pnts_to_allen_helper_func(np.load(converted_points), tifffile.imread(ann_pth), order = 'ZYX')
        
        #zmd added 20190312 - these should be in order of points inputted from raw space
        np.save(os.path.join(dst, "annotation_pixel_value_coordinates.npy"), point_lst)
        
        df = count_structure_lister(id_table, *point_lst).fillna(0)
        df.to_csv(os.path.join(dst, os.path.basename(id_table).replace('.xlsx', '')+'_with_anatomical_assignment_of_cell_counts.csv'))
            
        #load and convert to single voxel loc
        zyx = np.asarray([str((int(xx[0]), int(xx[1]), int(xx[2]))) for xx in load_np(converted_points)])
        zyx_cnt = Counter(zyx)
        
        #manually call transformix..
        transformed_dst = os.path.join(dst1, 'transformed_points'); makedir(transformed_dst)
        if qc_overlay_transform_type == 'all':
            tp0 = [xx for xx in listall(os.path.dirname(cellvol.ch_to_reg_to_atlas), 'TransformParameters.0.txt') if 'sig_to_reg' in xx and 'regtoatlas' not in xx][0]
            tp1 = [xx for xx in listall(os.path.dirname(cellvol.ch_to_reg_to_atlas), 'TransformParameters.1.txt') if 'sig_to_reg' in xx and 'regtoatlas' not in xx][0]
            transformfiles = [tp0, tp1, os.path.join(fld, 'elastix/TransformParameters.0.txt'), os.path.join(fld, 'elastix/TransformParameters.1.txt')]
        elif qc_overlay_transform_type == 'single':
            transformfiles = [os.path.join(fld, 'elastix/TransformParameters.0.txt'), os.path.join(fld, 'elastix/TransformParameters.1.txt')]
        elif qc_overlay_transform_type == 'affine_only_reg_to_sig':    
            tp0 = [xx for xx in listall(os.path.dirname(cellvol.ch_to_reg_to_atlas), 'TransformParameters.0.txt') if 'sig_to_reg' in xx and 'regtoatlas' not in xx][0]
            transformfiles = [tp0, os.path.join(fld, 'elastix/TransformParameters.0.txt'), os.path.join(fld, 'elastix/TransformParameters.1.txt')]
        transformfiles = modify_transform_files(transformfiles, dst = dst1)
        
        #cell_registered channel
        if generate_registered_overlay:
            transformix_command_line_call(cellvol.resampled_for_elastix_vol, transformed_dst, transformfiles[-1])
            cell_reg = tifffile.imread(os.path.join(transformed_dst, 'result.tif'))
            cell_cnn = np.zeros_like(cell_reg)
            errors = []        
            for zyx,v in zyx_cnt.items():
                z,y,x = [int(xx) for xx in zyx.replace('(','',).replace(')','').split(',')]
                try:
                    cell_cnn[z,y,x] = v*100
                except Exception as e:
                    print(e)
                    errors.append(e)
            if len(errors)>0:
                with open(os.path.join(dst, '{}_errors.txt'.format(os.path.basename(fld))), 'a') as flll:
                    for err in errors:
                        flll.write(str(err)+'\n')
                    flll.close()
            merged = np.stack([cell_cnn, cell_reg, np.zeros_like(cell_reg)], -1)
            #reorient to horizontal
            merged = np.swapaxes(merged, 0,2)
            tifffile.imsave(os.path.join(dst, 'generate_registered_overlay_{}_points_merged.tif'.format(os.path.basename(fld))), merged)
        if cleanup: shutil.rmtree(dst0)
        
        with open(os.path.join(dst, 'info.txt'), 'w') as fl:
            fl.write('FILES AND PATHS USED TO GENERATE ANALYSIS:\n')
            fl.write('\n')
            fl.write('path_to_csv_file: {}\n'.format(path_to_csv_file))
            fl.write('mat_file_suffix: {}\n'.format(mat_file_suffix))
            fl.write('dst_suffix: {}\n'.format(dst_suffix))
            fl.write('ann_pth: {}\n'.format(ann_pth))
            fl.write('id_table: {}\n'.format(id_table))
            fl.write('dst: {}\n'.format(dst))
            fl.write('fld: {}\n'.format(fld))
            fl.write('cell_centers_path: {}\n'.format(cell_centers_path))
            fl.write('\n')
            fl.write('\n')
            fl.write('Date code was run: {}\n'.format(datetime.datetime.today().strftime('%Y-%m-%d')))
            fl.close()
            