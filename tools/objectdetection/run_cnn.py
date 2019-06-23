#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:32:00 2018

@author: tpisano

Script to wrap kannanuv's  (github.com/kannanuv) CNN
"""

import os, sys, numpy as np, subprocess as sp, time, shutil, multiprocessing as mp
from tools.utils.io import load_kwargs, makedir, listdirfull, chunkit

if __name__ == '__main__':
    #manually apply CNN
    src = '/home/wanglab/Downloads/test_cnn_input' #folder of 16bit tiffs
    dst = '/home/wanglab/Downloads/test_cnn_output' #folder to save. Output of 8bit tiffs with background value of 1, and cells have values of 2
    apply_cnn_to_folder(src, dst, cores = 1)
       
    #using package:
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr'
    apply_cnn_to_folder_wrapper(src, cores = 1)
    
#%%
def apply_cnn_to_folder_wrapper(src=False, **kwargs):
    '''Function to take in pth to brain folder, and run CNN on cell channels, and postprocess
    
    src = destination of package (main folder)
    
    Optional:
    -----------
    matlab_file - path to predictSemanticLabels.m, if false assumes: 'lightsheet/tools/objectdetection/cellCounting/predictSemanticLabels.m'
    cnn_weights - path to network weights, if false assumes 'lightsheet/supp_files/h129_cnn.mat'
    '''   
    cores = kwargs['cores'] if 'cores' in kwargs else 1
    if cores>4: cores =4 #matlab seems to have issues with anyhting more than 4 cores?
    
    if src: kwargs.update(load_kwargs(src))
    if not src: kwargs.update(load_kwargs(**kwargs))
    matlab_file = kwargs['matlab_file'] if 'matlab_file' in kwargs else False
    cnn_weights = kwargs['cnn_weights'] if 'cnn_weights' in kwargs else False
    for cellch in [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch']:
        dst = os.path.join(src, 'cells', os.path.basename(cellch.full_sizedatafld_vol))
        
        if cores == 1: 
            apply_cnn_to_folder_helper(cellch.full_sizedatafld_vol, dst, matlab_file=matlab_file, cnn_weights=cnn_weights)

        elif cores>1:
            fls = listdirfull(src, keyword='tif'); fls.sort()
            p = mp.Pool(cores)
            iterlst = [(src, dst, matlab_file, cnn_weights, fls, core, cores) for core in range(cores)]
            p.map(apply_cnn_to_folder_par, iterlst)
            p.terminate()
    
    return

def apply_cnn_to_folder(src, dst, matlab_file=False, cnn_weights=False, cores=1):
    '''Function to apply cnn to a folder
    
    src = path to folder of tiffs
    dst = location to save cnn predictions
    
    Optional:
    -----------
    matlab_file - path to predictSemanticLabels.m, if false assumes: 'lightsheet/tools/objectdetection/cellCounting/predictSemanticLabels.m'
    cnn_weights - path to network weights, if false assumes 'lightsheet/supp_files/h129_cnn.mat'
    
    Cores = for parellization
    '''
    st = time.time()
    if cores>4: cores =4 #matlab seems to have issues with anyhting more than 4 cores?
    if cores == 1: 
        apply_cnn_to_folder_helper(src, dst, matlab_file, cnn_weights)
        sys.stdout.write('Completed CNN detection of {} in {} minutes'.format(os.path.basename(src), np.round(((time.time()-st)/60), 2)))
        
    elif cores>1:
        fls = listdirfull(src, keyword='tif'); fls.sort()
        p = mp.Pool(cores)
        iterlst = [(src, dst, matlab_file, cnn_weights, fls, core, cores) for core in range(cores)]
        p.map(apply_cnn_to_folder_par, iterlst)
        p.terminate()
    
    return



def apply_cnn_to_folder_par((src, dst, matlab_file, cnn_weights, fls, core, cores)):
    '''Function that takes a list of folders, moves them into subfolders for CNN parallelization
    '''
    #move files into subfolder
    fls = fls[slice(*chunkit(core, cores, fls))]
    src0 = os.path.join(src, 'job_{}'.format(core)); makedir(src0)
    [shutil.move(xx, os.path.join(src0, os.path.basename(xx))) for xx in fls]
    
    #apply cnn to subfolder
    apply_cnn_to_folder_helper(src0, dst, matlab_file=matlab_file, cnn_weights=cnn_weights)
    
    #move back out of subfolder
    [shutil.move(xx, os.path.join(src, os.path.basename(xx))) for xx in listdirfull(src0)]
    shutil.rmtree(src0)
    
    return



def apply_cnn_to_folder_helper(src, dst, matlab_file=False, cnn_weights=False):
    '''
    Saves planes in dst. Note that background is pixel value of 1, cells are a value of 2.
    
    matlab -nodisplay -nodesktop -r "addpath(genpath('/home/wanglab/wang/pisano/Python/lightsheet/tools/objectdetection')); predictSemanticLabels('/home/wanglab/Downloads/test_cnn_input', '/home/wanglab/wang/pisano/Python/lightsheet/supp_files/h129_cnn.mat', '/home/wanglab/Downloads/test_cnn_output'); quit"

    '''
    assert src != dst, 'src must not be the same as dst'
    if not cnn_weights: cnn_weights = os.path.join(os.getcwd(), 'supp_files/h129_cnn.mat')
    if not matlab_file: matlab_file = os.path.join(os.getcwd(), 'tools/objectdetection/cellCounting/predictSemanticLabels.m')
    makedir(dst)
    
    #setup commandline and call
    call0 = '''matlab -nodisplay -nodesktop -r "addpath(genpath('{}')); {}('{}', '{}', '{}'); quit"'''.format(os.path.dirname(matlab_file), os.path.splitext(os.path.basename(matlab_file))[0], src, cnn_weights, dst)
    sp_call(call0)
    
    return

def sp_call(call):
    '''Command line function and return output
    '''    
    from subprocess import check_output
    print check_output(call, shell=True)
    return 

def fix_folders(lst):
    '''uncollates folders in case apply_cnn_to_folder_par crashes
    '''
    for src in lst:
        [shutil.move(xx, os.path.dirname(src)) for xx in listdirfull(src)]
        [shutil.rmtree(src)]
    return