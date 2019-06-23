#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:09:00 2017

@author: tpisano
"""

if __name__ == '__main__':
    #example using post-registered elastix file
    src = '/jukebox/wang/pisano/Python/lightsheet/supp_files/sample_coordinate_to_location.xlsx'
    dst = '/home/wanglab/Downloads/sample_coordinate_to_location_output.xlsx'
    find_location(src, correspondence_type = 'post_elastix', dst=dst)

    #example using full size data
    src = '/home/wanglab/Downloads/coord_to_location_test.xlsx'
    dst = '/home/wanglab/Downloads/coord_to_location_test_output.xlsx'
    find_location(src, correspondence_type = 'full_size_data', verbose=True, dst=dst)
    


#%%
def find_location(src, dst=False, correspondence_type = 'post_elastix', verbose=False):
    '''
    Function to transform an excel sheet (e.g.: lightsheet/supp_files/sample_coordinate_to_location.xlsx) and output transformed locations.
    
    Suggestion is to use imagej to find XYZ coordinates to input into excel sheet.
        
    Inputs
    ----------------
    src = excelsheet
    correspondence_type = 
                    'post_elastix': your coordinates are the corresponding post-registered elastix file (outputfolder/elastix/..../result....tif)
                    'full_size_data': you coordinates are from the "full_sizedatafld" where:
                        Z== #### in 'file_name_Z####.tif'
                        X,Y are the pixel of that tif file
    
    
    Returns
    ----------------
    dst = (optional) output excelfile. Ensure path ends with '.xlsx'
    
    '''
    #from __future__ import division
    #import shutil, os, tifffile, cv2, numpy as np, pandas as pd, sys, SimpleITK as sitk
    #from tools.utils.io import listdirfull, load_kwargs, writer, makedir
    #from tools.conv_net.read_roi import read_roi, read_roi_zip
    from tools.registration.register import transformed_pnts_to_allen_helper_func
    from tools.registration.transform import structure_lister
    from tools.utils.io import load_kwargs, listdirfull, listall
    import SimpleITK as sitk
    import pandas as pd, numpy as np, os
    from skimage.external import tifffile
    
    if correspondence_type == 'post_elastix':
        print('This function assumes coordinates are from the corresponding post-registered elastix file. \nMake sure the excel file has number,<space>number,<space>number and not number,number,number')
        
        #inputs
        df = pd.read_excel(src)
        
        for brain in df.columns[1:]:
            print brain
            
            #load and find files
            kwargs = load_kwargs(df[brain][df['Inputs']=='Path to folder'][0])
            ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile']))
                   
            #Look up coordinates to pixel value
            xyz_points = np.asarray([(int(xx.split(',')[0]), int(xx.split(',')[1]), int(xx.split(',')[2])) for xx in df[brain][3:].tolist()])
            xyz_points = transformed_pnts_to_allen_helper_func(xyz_points, ann=ann, order='XYZ')
            
            #pixel id to transform
            if 'allen_id_table' in kwargs: 
                structures = structure_lister(pd.read_excel(kwargs['allen_id_table']), *xyz_points)
            else:
                structures = structure_lister(pd.read_excel(kwargs['volumes'][0].allen_id_table), *xyz_points)
            
            #update dataframe
            df[brain+' point transform'] = df[brain][:3].tolist()+[str(s.tolist()[0]) for s in structures]
    
            
        
        if not dst: dst = src[:-5]+'_output.xlsx'
        df.to_excel(dst)
        print('Saved as {}'.format(dst))
    
    if correspondence_type == 'full_size_data':
        from tools.imageprocessing.orientation import fix_dimension_orientation, fix_contour_orientation
        from tools.utils.directorydeterminer import pth_update
        from tools.registration.register import transformix_command_line_call, transformed_pnts_to_allen, collect_points_post_transformix
        from tools.registration.transform import points_resample, points_transform
        print('This function assumes coordinates are from the corresponding "full_sizedatafld". \nMake sure the excel file has number,<space>number,<space>number and not number,number,number')
        
        #inputs
        df = pd.read_excel(src)
        
        for brain in df.columns[1:]:
            print brain
            
            #load and find files
            kwargs = load_kwargs(df[brain][df['Inputs']=='Path to folder'][0])
            ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile']))
            ch_type = str(df[brain][df['Inputs']=='Channel Type'].tolist()[0])
            vol = [xx for xx in kwargs['volumes'] if xx.ch_type == ch_type][0]
                   
            #Look up coordinates to pixel value
            zyx_points = np.asarray([(int(xx.split(',')[2]), int(xx.split(',')[1]), int(xx.split(',')[0])) for xx in df[brain][3:].tolist()])
            
            #Fix orientation
            zyx_points = fix_contour_orientation(np.asarray(zyx_points), verbose=verbose, **kwargs)
            
            #Fix Scaling
            trnsfmdpnts = points_resample(zyx_points, original_dims=fix_dimension_orientation(vol.fullsizedimensions, **kwargs), resample_dims=tifffile.imread(pth_update(vol.resampled_for_elastix_vol)).shape, verbose=verbose)
            
            #write out points for transformix
            transformfile = [xx for xx in listall(os.path.join(vol.inverse_elastixfld)) if os.path.basename(vol.full_sizedatafld_vol)[:-5] in xx and 'atlas2reg2sig' in xx and 'reg2sig_TransformParameters.1.txt' in xx][0]
            tmpdst = os.path.join(os.path.dirname(src), 'coordinate_to_location_tmp')
            output = points_transform(src=trnsfmdpnts[:,:3], dst=tmpdst, transformfile=transformfile, verbose=True)
 
            #collect from transformix
            xyz_points = collect_points_post_transformix(output)
            
            #now ID:
            pix_ids = transformed_pnts_to_allen_helper_func(xyz_points, ann=ann, order='XYZ')
            
            #pixel id to transform
            aid = kwargs['allen_id_table'] if 'allen_id_table' in kwargs else kwargs['volumes'][0].allen_id_table
            structures = structure_lister(pd.read_excel(aid), *pix_ids)
            
            #update dataframe
            df[brain+' xyz points atlas space'] = df[brain][:3].tolist()+[str(s.tolist()[0]) for zyx in xyz_points]
            df[brain+' structures'] = df[brain][:3].tolist()+[str(s.tolist()[0]) for s in structures]
    
            
        
        if not dst: dst = src[:-5]+'_output.xlsx'
        df.to_excel(dst)
        print('Saved as {}'.format(dst))
    
    
    return

#############IGNORE BELOW

if False:
    fl='/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/full_sizedatafld/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_ch01/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_C01_Z1920.tif'
    transformfile='/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/elastix_inverse_transform/cellch_20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_resized_ch01_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.1.txt'
    atlas = '/home/wanglab/wang/pisano/Python/allenatlas/average_template_25_sagittal.tif'
    #
    #zyx_list=[(1920, 326,322), (1920, 305,314), (1920, 1333, 1020), (1920, 1367, 1020), (1920, 1350, 1000), (1920, 1350, 1030)]
    zyx_list=[(1920, 1350, 1020)]
    #%%
    #extract ROIs, fill contours, find nonzero pixels
    im = tifffile.imread(fl)
    #rois = read_roi_zip(roipth)
    #blnk = np.zeros((im.shape))#.astype('uint32')
    #for roi in rois:
    #    cv2.fillPoly(blnk, [np.int32(roi)], 255)
    #plt.ion(); plt.figure(); plt.imshow(blnk[0])
    
    #fix orientations
    from tools.imageprocessing.orientation import fix_contour_orientation, fix_dimension_orientation
    zyx_list = fix_contour_orientation(np.asarray(zyx_list), **kwargs)
    z,y,x=fix_dimension_orientation(kwargs['volumes'][0].fullsizedimensions, **kwargs)
    
    #account for point transform due to resizing      
    nx4centers=np.ones((len(zyx_list), 4))
    nx4centers[:,:-1]=zyx_list #inv
    zr, yr, xr = tifffile.imread(pth_update(kwargs['volumes'][0].resampled_for_elastix_vol)).shape
    trnsfrmmatrix=np.identity(4)*(zr/z, yr/y, xr/x, 1) ###downscale to "resampledforelastix size"
    #nx4 * 4x4 to give transform
    trnsfmdpnts=nx4centers.dot(trnsfrmmatrix) ##z,y,x
           
           
    #%%
    #write out points for transformix
    txtflnm='rois_xyz'
    os.remove(pth+'/'+txtflnm)
    #%%
    writer(pth, 'index\n{}\n'.format(len(trnsfmdpnts)), flnm=txtflnm)    
    sys.stdout.write('\nwriting centers to transfomix input points text file...')
    stringtowrite = '\n'.join(['\n'.join(['{} {} {}'.format(i[2], i[1], i[0])]) for i in trnsfmdpnts]) ####this step converts from zyx to xyz*****
    writer(pth, stringtowrite, flnm=txtflnm)
          
    #fix cluster pth:
    from tools.utils.update import search_and_replace_textfile
    for fl in ['/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/elastix_inverse_transform/cellch_20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_resized_ch01_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.0.txt', '/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/elastix_inverse_transform/cellch_20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_resized_ch01_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.1.txt', '/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/elastix_inverse_transform/cellch_20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_resized_ch01_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.0.txt', '/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/elastix_inverse_transform/cellch_20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec/20170118_bene_datcretransgen_2m_488_647_0010na_1hfsds_z3um_200msec_resized_ch01_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.1.txt']:
            search_and_replace_textfile(fl, '/jukebox', '/home/wanglab')
    
    #run transformix on points
    points_file = point_transfomix(pth+'/'+txtflnm, transformfile, chtype = 'injch')
    
    with open(points_file, 'r') as f:
            lines=f.readlines()
            f.close()
    arr = np.zeros((len(lines), 3))
    for i in range(len(lines)):        
            arr[i,...]=lines[i].split()[lines[i].split().index('OutputPoint')+3:lines[i].split().index('OutputPoint')+6] #x,y,z
    
    ann=sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile']))) ###zyx
    pnts=transformed_pnts_to_allen_helper_func(arr, ann)
    df=pd.read_excel('/home/wanglab/temp_wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')    
    df[df.id==pnts[-1].astype('int')]
            
    #convert registered points into structure counts
    transformed_pnts_to_allen(points_file, ch_type = 'injch', point_or_index=None, allen_id_table_pth='/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/lightsheet/supp_files/allen_id_table.xlsx', **kwargs)    
    #%%
    
    
    #%%
    from tools.registration.allen_structure_json_to_pandas import isolate_and_overlay, overlay
    structs=[]
    for i in df[df.id==pnts[-1].astype('int')].iterrows():
       structs.append(i[1][2])
    
    pth='/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
    AtlasFile='/home/wanglab/wang/pisano/Python/allenatlas/average_template_25_sagittal.tif'
    svlc='/home/wanglab/LightSheetData/witten-mouse/20170118_bene_datcretransgen_2m/overlays'
    makedir(svlc)
    #make allen structure overlay
    nm_to_sv='struct_overlay'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *structs)
    #make sphere of roi region
    nm_to_sv1='roi_overlay'
    z,y,x=(456, 528, 320)
    narr = np.asarray([(zz,yy,abs(xx-x)) for xx,yy,zz in arr])
    overlay(AtlasFile, svlc, nm_to_sv1, narr, gaussianblur=True) #from arr of xyz to np zyx
    #%%
    atl = tifffile.imread(atlas)
    structtif=tifffile.imread(svlc+'/'+nm_to_sv+'/'+nm_to_sv+'.tif')
    roitif=tifffile.imread(svlc+'/'+nm_to_sv1+'/'+nm_to_sv1+'.tif')
    z,y,x = atl.shape
    fl = np.zeros((z,y,x,3))
    fl[...,0]=structtif
    fl[...,1]=atl
    #this will be blue point coors
    fl[...,2]= roitif
    tifffile.imsave(svlc+'/'+'/roi_atlas_overlay.tif', fl.astype('uint8'))
