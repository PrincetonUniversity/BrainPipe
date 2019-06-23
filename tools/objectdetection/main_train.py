#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:39:41 2018

@author: tpisano
"""
from tools.objectdetection.train_random_forest import bin_data, train_random_forest, plot_roc, apply_random_forest, remove_border, zero_border
from tools.utils.io import makedir, chunkit, load_np, load_dictionary, save_dictionary, listdirfull, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer as dd, pth_update
import numpy as np, multiprocessing as mp,os, pandas as pd, gc,sys
from tools.conv_net.functions.bipartite import pairwise_distance_metrics_multiple_cutoffs
from skimage.external import tifffile
from tools.objectdetection.run_cnn import apply_cnn_to_folder
from tools.objectdetection.postprocess_cnn import load_tiff_folder, filter_size
from tools.objectdetection.evaluate_performance import compute_p_r_f1, visualize_centers, paired_centers, visualize_random_forests, plot_confusion_matrix
from skimage.morphology import closing, ball
from sklearn.metrics import confusion_matrix


#Test estimator and depth of random forest... using param sweep
#It's missing a lot on the cells on the edge - won't be the case for real testing. Consider padding these for performance metrics <- false, performance tanks with pad=true


if __name__ == '__main__':

    #rerun annotations (pth, src, dst)
    #run '/home/wanglab/wang/pisano/Python/lightsheet/tools/conv_net/input/input_data.py' '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_tp8' '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays'
    
    training_data_src = '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays'
    cnn_input = training_data_src+'_cnn_input'; makedir(cnn_input)
    cnn_output = training_data_src+'_cnn_output'; makedir(cnn_output)
    
    #apply cnn
    for ii,pth in enumerate(os.listdir(training_data_src)):
        #load, save out, and run cnn, and filter size
        arr = load_np(os.path.join(training_data_src, pth))[0]
        dst0 = os.path.join(cnn_input, pth[:-4])
        dst1 = os.path.join(cnn_output, pth[:-4])
        if not os.path.exists(dst0): 
            makedir(dst0); [tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), im) for i,im in enumerate(arr)]
        if not os.path.exists(dst1): 
            makedir(dst1); apply_cnn_to_folder(src=dst0, dst=dst1)
    
    #adjust size
    
    #Parameters for random forest
    size = (3,12,12) #(7,25,25) #(7,50,50) #Size: of cube around each center to train on.
    maxip=10 #Currently number of nonnormalized Maxip ravels to add to data
    cutoff = 10 #Cutoff: allowable euclidean distance (IN PIXELS) for a True positive
    n_estimator = 2000 #50 number of trees -ranodm search shows 800
    max_depth = 500 #depth of forest 50
    cores = 12 #for parallelization
    kfold_splits = 5 #10 # number of times to iterate through using kfold cross validation
    balance = False #optionally balance the number of each case (might help prevent biasing of "noncells"), better to use class_weight
    pad = False #important for edge cases in training set (i.e. points that don't have sufficient border around them)
    dst = False #'/home/wanglab/wang/pisano/Python/lightsheet/supp_files/h129_rf_classifier' #place to save classifier
    collect_cnn = True #optional to use CNN as input to forest in addition to the raw signal <-THIS IS ACTUALLY A SECOND COPY OF THE RAW DATA, SEEMS TO WORK BETTER
    max_features = None#'auto' #Slightly better with none, but muhc longer training time, None = number of features, auto=sqrt(#samples)
    class_weight = 'balanced' #None, 'balanced', 'balanced_subsample', auto' {0:.1, 1:.9} #if using dictionary might need to sum to 1
    verbose = True
    warm_start = True
    precision_score = None#'binary' #'weighted'
        #If None, the scores for each class are returned
        #'binary':Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
        #'micro':Calculate metrics globally by counting the total true positives, false negatives and false positives.
        #'macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        #'weighted':Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
        #'samples':Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    
    ##ADD Min_sampoles leaf = 1
    
    #NORM =TRUE
               
    #parse data
    dct = bin_data(cnn_output, training_data_src, size=size, cutoff = cutoff, pad = pad, verbose=verbose, collect_cnn = collect_cnn, maxip=maxip)
    #save_dictionary('/home/wanglab/wang/pisano/conv_net/annotations/better_res/dct_size_{}_{}_{}.p'.format(size[0], size[1], size[2]), dct)
    tps = dct['true_positives']; fps = dct['false_positives']; gts = dct['ground_truths'] #sitk.Show(sitk.GetImageFromArray(tps[48].reshape(7,25,25)))

    #train
    kwargs = train_random_forest(tps, fps, n_estimator = n_estimator, max_depth = max_depth, balance = balance, cores = cores, kfold_splits = kfold_splits, dst = dst, class_weight=class_weight, max_features=max_features, precision_score=precision_score, warm_start=warm_start)   
    if False: save_dictionary('/home/wanglab/wang/pisano/figures/cell_detection/randomforest/train_random_forest_dictionary.p', kwargs)
    
    #plot
    #%matplotlib inline
    save = '/home/wanglab/wang/pisano/figures/cell_detection/randomforest/rf'
    plot_roc(save=save, **kwargs)
    if False: visualize_random_forests(kwargs['classifier'], dst='/home/wanglab/wang/pisano/figures/cell_detection/randomforest/trees')
    
    # Compute confusion matrix
    save = '/home/wanglab/wang/pisano/figures/cell_detection/randomforest/confusion_matrix'
    fig = plot_confusion_matrix(cm = confusion_matrix(kwargs['y_test'], kwargs['classifier'].predict(kwargs['X_test'])), classes=['Not cell', 'Cell'], title='Confusion matrix, with normalization', dst=save+'_norm')
    fig = plot_confusion_matrix(cm = confusion_matrix(kwargs['y_test'], kwargs['classifier'].predict(kwargs['X_test'])), classes=['Not cell', 'Cell'], title='Confusion matrix, without normalization', normalize=False, dst=save+'_nonorm')
      
    #apply
    idx = 7
    cnn_src = listdirfull(cnn_output); cnn_src.sort(); cnn_src = cnn_src[idx]
    save = '/home/wanglab/wang/pisano/figures/cell_detection/randomforest/{}'.format(os.path.basename(cnn_src))
    inn = listdirfull(training_data_src); inn.sort(); 
    raw_src = load_np(inn[idx])[0]
    gt = load_np(inn[idx])[1]
    centers = apply_random_forest(kwargs['classifier'], size = size, raw_src = raw_src, cnn_src = cnn_src, cores = 10, pad=pad, chunks=1)
        
    #show
    src = np.zeros_like(gt)
    for c in centers.keys():
        src[c[0],c[1],c[2]] = 1

    #look at centers, removing detected centers and gts that are in the size border since this is unfair...
    gt_pnts = np.asarray(np.nonzero(remove_border(gt,size))).T
    detected_pnts = np.asarray(np.nonzero(remove_border(src,size))).T  
    visualize_centers([gt_pnts, detected_pnts], labels=['groundtruth', 'centers'], dst=save)
    
    #pair off and look at centers
    paired, tp, fp, fn = pairwise_distance_metrics_multiple_cutoffs(gt_pnts, detected_pnts, verbose=False, return_paired=True, cutoffs=[cutoff])[0]
    paired_centers(gt_pnts, detected_pnts, paired, dst=save)
    p,r,f1 = compute_p_r_f1(tp=tp, fn=fn,fp=fp)
    
    #sitk
    from tools.conv_net.functions.dilation import dilate_with_element, ball
    cnn_src = listdirfull(cnn_output); cnn_src.sort(); cnn_src = cnn_src[idx]
    cnn_src = load_tiff_folder(cnn_src)
    gt = dilate_with_element(gt, ball(5))
    src = dilate_with_element(src, ball(5))
    src = zero_border(src, size)
    gt = zero_border(gt, size)
    sitk.Show(sitk.GetImageFromArray(raw_src))
    sitk.Show(sitk.GetImageFromArray(cnn_src))
    sitk.Show(sitk.GetImageFromArray(src))
    sitk.Show(sitk.GetImageFromArray(gt))
    
    #############################################
    #test train split - where classifier never sees ANY DATA
    #############################################
    cnn_output = listdirfull(cnn_output); cnn_output.sort()
    cnn_output_train = cnn_output[::2]
    cnn_output_test = cnn_output[1::2]
    training_data_src = listdirfull(training_data_src); training_data_src.sort();
    training_data_src_train = training_data_src[::2]
    training_data_src_test = training_data_src[1::2]

    #parse data
    dct = bin_data(cnn_output_train, training_data_src_train, size=size, cutoff = cutoff, pad = pad, verbose=verbose)
    tps = dct['true_positives']; fps = dct['false_positives']; gts = dct['ground_truths']
    
    #train
    kwargs = train_random_forest(tps, fps, n_estimator = n_estimator, max_depth = max_depth, cores = cores, kfold_splits = kfold_splits, dst = dst)   
    plot_roc(save=False, **kwargs)
    
    #eval
    lst = []; show=True
    for i, cnn_src in enumerate(cnn_output_test):
        inn = training_data_src_test; inn.sort(); inn = inn[i]
        raw_src = load_np(inn)[0]
        gt = load_np(inn)[1]
        gt_pnts = np.asarray(np.nonzero(remove_border(gt,size))).T
        centers = apply_classifier(kwargs['classifier'], raw_src, cnn_src, size = size, pad=pad)
        src = np.zeros_like(gt)
        for c in centers.astype('int'):
            src[c[0],c[1],c[2]] = 1
        gt_pnts = np.asarray(np.nonzero(remove_border(gt,size))).T
        detected_pnts = np.asarray(np.nonzero(remove_border(src,size))).T  
        if show: visualize_centers([gt_pnts, detected_pnts], labels=['groundtruth', 'centers'], dst=False)
        paired, tp, fp, fn = pairwise_distance_metrics_multiple_cutoffs(gt_pnts, detected_pnts, verbose=False, return_paired=True, cutoffs=[40])[0]
        if show: paired_centers(gt_pnts, detected_pnts, paired, dst=False)
        p,r,f1 = compute_p_r_f1(tp=tp, fn=fn,fp=fp)
        lst.append(tp,fp,fn,p,r,f1)
    
        from tools.conv_net.functions.dilation import dilate_with_element, ball
        cnn_src = load_tiff_folder(cnn_src)
        gt = dilate_with_element(gt, ball(5))
        src = dilate_with_element(src, ball(5))
        src = zero_border(src, size)
        gt = zero_border(gt, size)
        sitk.Show(sitk.GetImageFromArray(raw_src))
        sitk.Show(sitk.GetImageFromArray(cnn_src))
        sitk.Show(sitk.GetImageFromArray(src))
        sitk.Show(sitk.GetImageFromArray(gt))
    
    #from sklearn import cross_validation.KFold
    #from sklearn import confusion_matrix