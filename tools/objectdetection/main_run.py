#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:33:34 2018

@author: tpisano
"""
import os, sys, numpy as np, time
from tools.utils.io import load_kwargs, save_dictionary, makedir, listdirfull
from tools.objectdetection.run_cnn import apply_cnn_to_folder
from tools.objectdetection.random_forest import apply_classifier
from tools.utils.directorydeterminer import pth_update


if __name__ == '__main__':
    from tools.objectdetection.main_run import cell_detect_wrapper
    #manually apply CNN
    src = '/home/wanglab/Downloads/test_cnn_input' #folder of 16bit tiffs
    dst = '/home/wanglab/Downloads/test_cnn_output' #folder to save. Output of 8bit tiffs with background value of 1, and cells have values of 2
    apply_cnn_to_folder(src, dst)
        
    #using package:
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr' #great to demonstrate prevention of FPs
    src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02'
    cell_detect_wrapper(src, cores=4, numZSlicesPerSplit=12, overlapping_planes=12, classifier_window_size = (3,12,12), verbose = True)
    
    
    inn = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/test'
    out = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/test_out'
    outpth = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20170115_tp_bl6_lob6a_1000r_02/cells/test_out.p'
    #apply_cnn_to_folder(src=inn, dst=out, cores = 4)
    rf = kwargs['classifier']
    out = apply_classifier(rf, raw_src = inn, cnn_src = out, collect_cnn =True, size = (3,10,10), pad=False, cores=1, numZSlicesPerSplit=40, overlapping_planes = 1, verbose=True)
    save_dictionary(outpth, {k:v[1:] for k,v in out.iteritems()})
    from tools.objectdetection.detection_qc import overlay_cells
    vol, vol0 = overlay_cells(src, {k:v[1:] for k,v in out.iteritems()}, dilation_radius = 5, load_range = False)

#%%
def cell_detect_wrapper(src=False, **kwargs):
    '''Function to take in pth to brain folder, and run CNN on cell channels, and then run random forest classifier
    
    src = destination of package (main folder)
    '''
    #load
    if src: kwargs.update(load_kwargs(src))
    if not src: kwargs.update(load_kwargs(**kwargs))

    #run for each cellch
    for cellch in [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch']:
        cell_detect(src = cellch.full_sizedatafld_vol, dst = os.path.join(os.path.dirname(os.path.dirname(cellch.full_sizedatafld_vol)), 'cells', os.path.basename(cellch.full_sizedatafld_vol)), **kwargs)
    return


def cell_detect(src, **kwargs):
    '''Function to apply CNN to folder and then random forest classifier
    
    Inputs
    ---------
    src = path to tifffolder
    
    Optional
    --------------
    dst - location to save files
    #CNN inputs
        matlab_file - path to predictSemanticLabels.m, if false assumes: 'lightsheet/tools/objectdetection/cellCounting/predictSemanticLabels.m'
        cnn_weights - path to network weights, if false assumes 'lightsheet/supp_files/h129_cnn.mat'
    #Classifier Inputs
        classifier - path to .pkl of trained classifier
        classifier_window_size - size of window for classifier
        numZSlicesPerSplit: chunk of zplanes to process at once. Adjust this and cores based on memory constraints.
        cores: number of parallel jobs to do at once. Adjust this and numZSlicesPerSplit based on memory constraints
        overlapping_planes: number of planes on each side to overlap by, this should be comfortably larger than the maximum z distances of a single object
        pad = (optional) usually false, deals with how to handle edge cases (i.e. points that don't have sufficient border around them)
            True if pnt is on edge of image, function pads evenly
            False if pnt is on edge of image, drop. 
        collect_cnn = optional provides cnn as input to RF in addition to raw
        maxip = number of maxips to include in raveled data
        
    Defaults:
    #cnn inputs
        matlab_file = kwargs['matlab_file'] if 'matlab_file' in kwargs else False
        cnn_weights = kwargs['cnn_weights'] if 'cnn_weights' in kwargs else False
    #classifier inputs
        cores = kwargs['cores'] if 'cores' in kwargs else 1
        classifier = kwargs['classifier'] if 'classifier' in kwargs else os.path.join(os.getcwd(), 'supp_files/h129_rf_classifier.pkl')
        classifier_window_size = kwargs['classifier_window_size'] if 'classifier_window_size' in kwargs else (7,50,50)
        numZSlicesPerSplit = kwargs['numZSlicesPerSplit'] if 'numZSlicesPerSplit' in kwargs else 250
        overlapping_planes = kwargs['overlapping_planes'] if 'overlapping_planes' in kwargs else 40
        pad = kwargs['pad'] if 'pad' in kwargs else False
        chunks = kwargs['chunks'] if 'chunks' in kwargs else 50
        collect_cnn = kwargs['collect_cnn'] if 'collect_cnn' in kwargs else True
    
    If memory issues, best bet is to decrease cores, and numZSlicesPerSplit
    
    
    Saves:
        .p file 
    
    '''
    st = time.time()
    dst = kwargs['dst'] if 'dst' in kwargs else src
    #cnn inputs
    matlab_file = kwargs['matlab_file'] if 'matlab_file' in kwargs else False
    cnn_weights = kwargs['cnn_weights'] if 'cnn_weights' in kwargs else False
    #classifier inputs
    cores = kwargs['cores'] if 'cores' in kwargs else 1
    classifier = pth_update(kwargs['classifier']) if 'classifier' in kwargs else os.path.join(os.getcwd(), 'supp_files/h129_rf_classifier.pkl')
    classifier_window_size = kwargs['classifier_window_size'] if 'classifier_window_size' in kwargs else (3,12,12)
    numZSlicesPerSplit = kwargs['numZSlicesPerSplit'] if 'numZSlicesPerSplit' in kwargs else 12
    overlapping_planes = kwargs['overlapping_planes'] if 'overlapping_planes' in kwargs else 12
    pad = kwargs['pad'] if 'pad' in kwargs else False
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    collect_cnn = kwargs['collect_cnn'] if 'collect_cnn' in kwargs else True
    maxip = kwargs['maxip'] if 'maxip' in kwargs else 0
    
    #setup folders
    #makedir(dst)
    cnn_output = dst+'_cnn_output'; makedir(cnn_output)
    
    #run cnn, if it hasn't been run already
    if not len(listdirfull(src, keyword='.tif')) == len(listdirfull(cnn_output, keyword='.tif')):
        apply_cnn_to_folder(src, dst=cnn_output, matlab_file=matlab_file, cnn_weights=cnn_weights, cores=np.min((cores, 4)))
        sys.stdout.write('Completed CNN detection of {} in {} minutes'.format(os.path.basename(src), np.round(((time.time()-st)/60), 2)))
    
    #run classifier
    sys.stdout.write('Starting Random forest detection...'); sys.stdout.flush()
    out = apply_classifier(classifier, raw_src = src, cnn_src = cnn_output, collect_cnn = collect_cnn, size = classifier_window_size, pad=pad, cores=cores, numZSlicesPerSplit=numZSlicesPerSplit, overlapping_planes = overlapping_planes, verbose=verbose, maxip=maxip)
    sys.stdout.write('Completed Random forest detection of {}. {} minutes total'.format(os.path.basename(src), np.round(((time.time()-st)/60), 2)))
    
    #save out - with and without cnn_pixels
    save_dictionary(dst+'_centers_cnn_intensity_radius.p', out)
    save_dictionary(dst+'_centers_intensity_radius.p', {k:v[1:] for k,v in out.iteritems()})
    
    return
    