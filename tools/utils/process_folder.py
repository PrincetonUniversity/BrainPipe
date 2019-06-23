#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:56:39 2017

@author: tpisano
"""
import os
from tools.utils.io import listdirfull, makedir
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.process_local import run_brain_locally
#fls = listdirfull('/home/tpisano/wang/pisano/tracing_output/vr/raw_data')
#location of raw data:
fls = listdirfull('/home/tpisano/LightSheetTransfer/mk')
#output folder:
out = '/home/tpisano/wang/mkislin/Histology/light_sheet/201707_ofp_mini'; makedir(out)
#test name range:
name_range=[7,18]
print ('This file:\n   {}\nName of folder:\n  {}'.format(fls[0], os.path.basename(fls[0])[name_range[0]:name_range[1]]))


for pth in fls:
    try:    
        nm = os.path.basename(pth)[name_range[0]:name_range[1]]
    
        systemdirectory=directorydeterminer()
        inputdictionary={
        pth : [['regch', '00'], ['injch', '01']]} #'cellch' and the '##' corresponds to the order LVBT acquired the channel
        
        params={
        'labeltype': 'aav', #'h129', 'prv', 'cfos'
        'objectdetection': 'edgedetection', # 'edgedetection', 'convnet', 'clearmap', 'all'; clearmap setting uses their SpotDetection method
        'systemdirectory':  systemdirectory, #don't need to touch
        'inputdictionary': inputdictionary, #don't need to touch
        'outputdirectory': os.path.join(out, nm),
        #CHANGE ME:
        'xyz_scale': (5, 5, 10), #(5.0,5.0,3), #micron/pixel: 5.0um/pix for 1.3x; 1.63um/pix for 4x objective
        'tiling_overlap': 0.00, #percent overlap taken during tiling
        'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!
        'annotationfile' :   os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/annotation_25_ccf2015_forDVscans.nrrd'), ###path to annotation file for structures'blendtype' : 'sigmoidal', #False/None, 'linear', or 'sigmoidal' blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental
        'intensitycorrection' : True, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
        'resizefactor': 3, ##in x and y #normally set to 5 for 4x objective, 3 for 1.3x obj
        'rawdata' : True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
        'finalorientation' : ('2','1', '0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation; ('2','1','0') for horizontal to sagittal
        'slurmjobfactor': 3 #number of array iterations per arrayjob since max job array on SPOCK is 1000
        }    
    
        #run_brain_locally(cores=8, steps=[0,1,2,3,4,5], **params)
        run_brain_locally(cores=8, steps=[0,1,2,3], **params) #for injection site detection 0,1,2,3 are needed. NOT 4 and 5.
    except Exception, e:
        print ('{} failed. \n   Error: {}'.format(pth, e))
        
#%%
#zebrin
from tools.utils.io import listdirfull, load_kwargs, save_kwargs
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.process_local import run_brain_locally
from tools.utils.update import change_line_in_run_tracing_file
import shutil, os, sys
fls=[ '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_01',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_02',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_03',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_04',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_05',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_old_01',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_old_02',
     '/home/wanglab/wang/pisano/tracing_output/zebrin/20161228_tp_bl6_zebrin_old_03']
try:
    sys.path.remove('/home/wanglab/wang/pisano/Python/tracing')
except:
    pass

for pth in fls:
    print pth
    from tools.utils.update import update_lightsheet_folder
    update_lightsheet_folder(pth)
    #delete elastix folder
    try:
        shutil.rmtree(pth+'/maskedatlas')
        shutil.rmtree(pth+'/elastix')
        shutil.rmtree(pth+'/elastix_inverse_transform')
        shutil.rmtree(pth+'/cells')
        shutil.rmtree(pth+'/injection')
    except:
        pass
    
    #fix run_tracing
    npth = pth+'/lightsheet'
    os.chdir(npth)
    npth = npth+'/run_tracing.py'
    sys.path.insert(0, npth)

    #original_text = ["'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_10_cb_posterior_to_anterior.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!", "'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_10_cb_posterior_to_anterior.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!"]
    #new_text = "'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/atlas/sagittal_atlas_20um_iso.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!"
    #for orig in original_text:
    #   change_line_in_run_tracing_file(pth, original_text=orig, new_text=new_text)

    #setup cropping: #ONLY DONE ONCE
    #original_text = "'maskatlas': {'x': all, 'y': '425:', 'z': all}"
    #original_text ="'maskatlas': {'x': all, 'y': '425:', 'z': all}"
    #new_text ="'cropatlas': {'x': all, 'y': '425:', 'z': all}\n}\n"
    #change_line_in_run_tracing_file(pth, original_text=original_text, new_text=new_text)
    
    #orig="'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/atlas/sagittal_atlas_20um_iso.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!"
    #new = "'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/atlas/sagittal_atlas_20um_iso.tif'), ###it is assumed that input image will be a horizontal scan with anterior being 'up'; USE .TIF!!!!\n'atlas_scale': (20, 20, 20), #micron/pixel, ABA is likely (25,25,25)"
    #change_line_in_run_tracing_file(pth, original_text=orig, new_text=new)

    #orig ="'finalorientation' : ('2','1', '0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation; ('2','1','0') for horizontal to sagittal"
    #new ="'finalorientation' : ('1','-2', '0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation; ('2','1','0') for horizontal to sagittal"
    #change_line_in_run_tracing_file(pth, original_text=orig, new_text=new)
    
    #run
    from tools.imageprocessing.preprocessing import updateparams
    from run_tracing import params
    params = updateparams(**params)
    save_kwargs(**params)    
    run_brain_locally(cores=8, steps=[1,2,3], save_before_reorientation=True, **params)
    sys.path.remove(npth)
