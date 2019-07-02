#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:38:18 2018

@author: wanglab
"""

import numpy as np
from tools.conv_net.utils.io import pairwise_distance_metrics, load_dictionary, read_roi_zip

def human_compare_with_raw_rois(ann1roipth, ann2roipth, cutoff = 30):
            
    #format ZYX, and remove any rois missaved
    ann1_zyx_rois = np.asarray([[int(yy) for yy in xx.replace(".roi", "").split("-")] for xx in read_roi_zip(ann1roipth, include_roi_name=True)])
    ann2_zyx_rois = np.asarray([[int(yy) for yy in xx.replace(".roi", "").split("-")] for xx in read_roi_zip(ann2roipth, include_roi_name=True)])
        
    paired,tp,fp,fn = pairwise_distance_metrics(ann1_zyx_rois, ann2_zyx_rois, cutoff) #returns true positive = tp; false positive = fp; false negative = fn
        
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    print ("\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
    Cutoff: {} \n\
    F1 score: {}% \n\
    true positives, false positives, false negatives: {} \n\
    precision: {}% \n\
    recall: {}%\n".format(cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))

    return tp, fp, fn, f1

if __name__ == "__main__":
    
    #load points dict
    points_dict = load_dictionary("/home/wanglab/Documents/cfos_inputs/cfos_points_dictionary.p")   
        
    print(points_dict.keys())
    #separate annotators - will have to modify conditions accordinaly
    ann1_dsets = ["tp_ann_201904_an22_ymazefos_020719_pfc_z150-169.npy",
                  "tp_ann_201904_an30_ymazefos_020719_striatum_z416-435.npy"]
#                 ["tp_ann_201904_an19_ymazefos_020719_pfc_z380-399.npy", 
#                  "tp_ann_201812_pcdev_lob6_9_forebrain_hypothal_z520-539.npy", 
#                  "tp_ann_201812_pcdev_lob6_4_forebrain_cortex_z200-219.npy"]

    ann2_dsets = ["jd_ann_201904_an22_ymazefos_020719_pfc_z150-169.npy",
                  "jd_ann_201904_an30_ymazefos_020719_striatum_z416-435.npy"]

    
    #initialise empty vectors
    tps = []; fps = []; fns = []   
    
    #set voxel cutoff value
    cutoff = 5
    
    for i in range(len(ann2_dsets)):
    
        #set ground truth
        print(ann1_dsets[i])
        ann1_ground_truth = points_dict[ann1_dsets[i]]
        ann2_ground_truth = points_dict[ann2_dsets[i]]
        
        paired,tp,fp,fn = pairwise_distance_metrics(ann2_ground_truth, ann1_ground_truth, cutoff = 30) #returns true positive = tp; false positive = fp; false negative = fn
        
        #f1 per dset
        precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
        f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
        tps.append(tp); fps.append(fp); fns.append(fn) #append matrix to save all values to calculate f1 score
        
    tp = sum(tps); fp = sum(fps); fn = sum(fns) #sum all the elements in the lists
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    print ("\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
    Cutoff: {} \n\
    F1 score: {}% \n\
    true positives, false positives, false negatives: {} \n\
    precision: {}% \n\
    recall: {}%\n".format(cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))
