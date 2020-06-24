#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:57:20 2018

@author: tpisano
"""

import os, sys, numpy as np, time, gc, copy, multiprocessing as mp, shutil
from tools.objectdetection.train_random_forest import get_pixels_around_center_mem_eff
from tools.utils.io import makedir, load_dictionary, save_dictionary, listdirfull
from scipy.ndimage.measurements import label, center_of_mass
from scipy.ndimage import find_objects, distance_transform_edt
from tools.objectdetection.postprocess_cnn import load_tiff_folder
from sklearn.externals import joblib


def apply_classifier(classifier, raw_src, cnn_src, collect_cnn = False, size = (3,12,12), pad=False, cores=10, numZSlicesPerSplit=50, overlapping_planes = 15, verbose=True, save=True, maxip=0):
    '''
    classifier = pretrained random forest or path to pretrained random forest
    raw_src = folder of tiffs of raw input data
    cnn_src = folder of tiffs from cnn output
    size
    pad = if True, pad the edges of objects determined. False: remove edge cases, usually better since they will be few in number
    cores = number of cores for parallelization, larger the number the less memory efficient
    numZSlicesPerSplit: chunk of zplanes to process at once. Adjust this and cores based on memory constraints.
    overlapping_planes: number of planes on each side to overlap by, this should be a comfortable amount larger than the maximum z distances of a single object
    save (optional): #optional save to prevent rerun of jobs
    collect_cnn = optional to include cnn data for random forest input
    
    Returns
    ----------------
    a dictionary consisting of k=centers, v=[corresponding pixel indices determine by CNN, maximum intensity, list of maximum radius/plane]
    
    '''
    #handle inputs
    threshold = 1
    zyx_search_range = (2,10,10)
    zdim = len(listdirfull(cnn_src, keyword='.tif'))
    
    #optional save to prevent rerun of jobs
    if save: 
        save = cnn_src+'_apply_classifier_tmp'
        makedir(save)
    
    #run
    if verbose: sys.stdout.write('\n   Thesholding, determining connected pixels, identifying center of masses, applying classifier\n\n'); sys.stdout.flush(); st = time.time()
    rng = range(0, zdim, numZSlicesPerSplit); jobs = len(rng);
    iterlst=[(cnn_src, raw_src, collect_cnn, z, zdim, numZSlicesPerSplit, overlapping_planes, threshold, classifier, size, zyx_search_range, pad, job, jobs, verbose, save, maxip) for job, z in enumerate(rng)]
    #par vs not par
    if cores > 1:
        p = mp.Pool(cores)
        center_pixels_intensity_radius_lst = p.starmap(apply_classifier_helper, iterlst)
        p.terminate()
    else:
        center_pixels_intensity_radius_lst = []
        for i in iterlst:
            center_pixels_intensity_radius_lst.append(apply_classifier_helper(i))
    #optional reload:
    if save:
        center_pixels_intensity_radius_lst = [load_dictionary(xx) for xx in listdirfull(save)]
        shutil.rmtree(save)

    #unpack
    if verbose: sys.stdout.write('\n...finished, formatting dictionary...'); sys.stdout.flush()
    center_pixels_intensity_radius_dct = {}; [center_pixels_intensity_radius_dct.update(xx) for xx in center_pixels_intensity_radius_lst]
    if 'None' in center_pixels_intensity_radius_dct: del center_pixels_intensity_radius_dct['None']
        
    if verbose: print ('Total time {} minutes'.format(round((time.time() - st) / 60)))
    if verbose: print('{} centers found.'.format(len(center_pixels_intensity_radius_dct)))

    return center_pixels_intensity_radius_dct

def apply_classifier_helper(cnn_src, raw_src, collect_cnn, z, zdim, numZSlicesPerSplit, overlapping_planes, threshold, classifier, size, zyx_search_range, pad, job, jobs, verbose, save, maxip):
    '''
    '''
    #don't run if already there....
    if save:
        if os.path.exists(os.path.join(save, '{}_{}.p'.format(job, jobs))):
            if verbose: sys.stdout.write('\n  Skipping job {} of {} as it already exists, delete if you would like to rerun'.format(job, jobs)); sys.stdout.flush()
            return {'None': 'None'}
        
    #else run
    assert overlapping_planes > 0, 'this function requires overlapping_planes to be greater than 0'
    #process
    st = time.time()
    if z == 0:
        arr = load_tiff_folder(cnn_src, threshold=threshold, load_range = (z, numZSlicesPerSplit+overlapping_planes))
        arr[arr>0]=1
        #find labels
        labels = label(arr)
        centers = center_of_mass(arr, labels[0], range(1, labels[1]+1)); 
        #convert to float16 for mem
        centers = [tuple((xx[0].astype('float16'), xx[1].astype('float16'), xx[2].astype('float16'))) for xx in centers]
        #return pixels associated with a center
        center_pixels_dct = return_pixels_associated_w_centers(centers, labels)
        center_radius_dct = find_radius(centers, labels); del labels, arr; gc.collect()
        #filter such that you only keep centers in first half
        centers = [center for center in centers if (center[0] <= numZSlicesPerSplit)]

    else: #cover 3x
        #find bounds since you are only loading a subset of volume
        zl = max(0, z - overlapping_planes)
        zadjusted = z - zl
        #load
        arr = load_tiff_folder(cnn_src, threshold=threshold, load_range = (z - overlapping_planes, z + numZSlicesPerSplit + overlapping_planes))
        arr[arr>0]=1
        zdim, ydim, xdim = arr.shape
        #find labels
        labels = label(arr)
        centers = center_of_mass(arr, labels[0], range(1, labels[1]))
        #convert to float16 for mem
        centers = [tuple((xx[0].astype('float16'), xx[1].astype('float16'), xx[2].astype('float16'))) for xx in centers]
        #return pixels associated with a center
        center_pixels_dct = return_pixels_associated_w_centers(centers, labels)
        center_radius_dct = find_radius(centers, labels); del labels, arr; gc.collect()
        #filter such that you only keep centers within middle third if overlapping_planes is greater than 0
        centers = [center for center in centers if (center[0] > zadjusted) and (center[0] <= np.min(((zadjusted+numZSlicesPerSplit), zdim)))]
        

    #adjust z plane to accomodate chunking. NOTE NOT ADJUSTING Z PLANE ON PIXELS - WILL DO AFTER FILTERING FOR EFFECIENCY
    center_pixels_dct = {tuple((c[0]+z, c[1], c[2])):center_pixels_dct[c] for c in centers}
    center_radius_dct = {tuple((c[0]+z, c[1], c[2])):center_radius_dct[c] for c in centers}; del centers
    
    #load and collect pixels
    if collect_cnn == False: cnn_src = False
    inn = get_pixels_around_center_mem_eff(pnts=center_pixels_dct.keys(), src=raw_src, cnn_src = cnn_src, size = size, pad=pad, return_pairs=True, cores=1, chunks=1, maxip=maxip)
    
    #edge cases
    if len(inn) == 0:
        save_dictionary(os.path.join(save, '{}_{}.p'.format(job, jobs)), {'None': 'None'})
        return 
    
    #predict, then remove centers that are considered false positives (where rf.predict outputs 0). Helps w/ memory as well
    rf = joblib.load(classifier) if type(classifier)==str else classifier
    cen_pred ={c:p for c,p in zip(inn.keys(), rf.predict(inn.values())) if p>0}
    valid_inn = {k:v for k,v in inn.iteritems() if k in cen_pred}
    center_pixels_dct = {k:v for k,v in center_pixels_dct.iteritems() if k in cen_pred}
    del inn; gc.collect()
    
    #no valid cells found
    if len(valid_inn) == 0:
        save_dictionary(os.path.join(save, '{}_{}.p'.format(job, jobs)), {'None': 'None'})
        return 
    
    #find max intensity given some window size
    cen_maxinten = {cen:find_intensity_at_center(pxl, zyx_search_range, size=size, collect_cnn=collect_cnn) for cen,pxl in valid_inn.iteritems()}
    
    #clean up radius dct by dropping centers not present
    center_radius_dct = {c:center_radius_dct[c] for c in center_pixels_dct.keys()}
        
    #form a dictionary k=centers, v = [cnn determined pixels of valid cells, maximum_intensity, [median radius, max radius]]. Also will now adjust the cnn determined pixels in z
    center_pixels_dct = {k:np.asarray((v[:,0]+z, v[:,1], v[:,2])).T for k,v in center_pixels_dct.iteritems()} 
    center_pixels_intensity_radius_dct = copy.deepcopy({c:[center_pixels_dct[c], cen_maxinten[c], center_radius_dct[c]] for c in valid_inn.keys()})
    
    #zero out, since error to delete
    cen_maxinten=0; valid_inn=0; center_pixels_dct=0; gc.collect()
    
    #convert keys to float16
    out = {tuple((k[0].astype('float16'),k[1].astype('float16'),k[2].astype('float16'))):v for k,v in center_pixels_intensity_radius_dct.iteritems()}
    
    #save
    if verbose: 
        sys.stdout.write('\n   Completed {} of {} in {} minutes, {} cells'.format(job+1, jobs, np.round((time.time() - st) / 60,2), len(out))); sys.stdout.flush()

    if save:
        save_dictionary(os.path.join(save, '{}_{}.p'.format(job, jobs)), out)
        return
        
    return out


def return_pixels_associated_w_centers(centers, labels):
    '''Function to return dictionary of k=centers, v=pixels of a given label
    ''' 
    dct = {}
    slices = find_objects(labels[0])
    for i,cen in enumerate(centers):
        z,y,x = [aa.astype('int') for aa in cen]
        sl = slices[i]
        #adjust in ZYX pixels for shift
        dct[cen] = np.asarray([tuple((aa[0]+int(sl[0].start), aa[1]+int(sl[1].start), aa[2]+int(sl[2].start))) for aa in np.asarray(np.where(labels[0][sl]==(i+1))).T])
    return dct

def find_intensity_at_center(src_raw, zyx_search_range=(2,10,10), size = (3,12,12), collect_cnn=False):
    '''function to return maximum intensity of center of src_raw given search range
    '''
    ln = src_raw.shape[0]
    if not collect_cnn: src_raw = src_raw.reshape(tuple([(2*xx+1) for xx in size]))
    if collect_cnn: src_raw = src_raw[:ln/2].reshape(tuple([(2*xx+1) for xx in size]))
    cen = [int(xx/2) for xx in src_raw.shape]
    rn = [slice(xx-yy, xx+yy+1) for xx,yy in zip(cen, zyx_search_range)]
    return np.max(src_raw[rn[0], rn[1], rn[2]])

def find_radius(centers, labels):
    '''Function to return dictionary of k=centers, v=list of maximum radius/plane
    ''' 
    dct = {}
    slices = find_objects(labels[0])
    for i,cen in enumerate(centers):
        z,y,x = [aa.astype('int') for aa in cen]
        sl = slices[i]
        arr = np.copy(labels[0][sl])
        arr[arr!=i+1] = 0
        dct[cen] = [np.max(xx).astype('float16') for xx in distance_transform_edt(arr)]
    return dct


if __name__ == '__main__':
    from tools.objectdetection.random_forest import apply_classifier
    #dst = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr/cells/20161214_db_bl6_crii_l_53hr_647_010na_z7d5um_75msec_5POVLP_ch00'
    classifier = '/home/wanglab/wang/pisano/Python/lightsheet/supp_files/h129_rf_classifier.pkl'
    cnn_src = '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr/cells/20161214_db_bl6_crii_l_53hr_647_010na_z7d5um_75msec_5POVLP_ch00_cnn_output'
    raw_src =  '/home/wanglab/wang/pisano/tracing_output/antero_4x/20161214_db_bl6_crii_l_53hr/full_sizedatafld/20161214_db_bl6_crii_l_53hr_647_010na_z7d5um_75msec_5POVLP_ch00'
    
    out = apply_classifier(classifier, raw_src, cnn_src, size = (7,50,50), pad=False, cores=10, numZSlicesPerSplit=25, overlapping_planes = 10, verbose=True)
    
    
    #check
    from tools.conv_net.functions.dilation import dilate_with_element, generate_arr_from_pnts_and_dims, ball
    pnts = [tuple((xx[0] - z, xx[1], xx[2])) for xx in out.keys()] #z in the from below eg 230
    arr1 = dilate_with_element(generate_arr_from_pnts_and_dims(pnts, dims = (40, 7422, 6262)), selem=ball(5))
