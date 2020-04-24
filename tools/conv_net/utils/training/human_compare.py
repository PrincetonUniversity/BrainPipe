#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:38:18 2018

@author: wanglab
"""

import numpy as np, pickle
from utils.io import pairwise_distance_metrics

if __name__ == "__main__":
    
    #load points dict
    pth = "/jukebox/wang/zahra/conv_net/annotations/prv/all/all_points_dictionary.p"
    points_dict = pickle.load(open(pth, "rb"), encoding = "latin1")
        
    print(points_dict.keys())
    total_cells = sum([len(v) for k,v, in points_dict.items()])
    print("\ntotal cells annotated: {}".format(total_cells))
     
    #separate annotators - will have to modify conditions accordinaly
    ann1_dsets = ["zd_ann_prv_jg05_neocortex_z310-449_01.npy", 
                  "zd_ann_prv_jg24_neocortex_z300-400_01.npy", 
                  "zd_ann_prv_jg29_neocortex_z300-500_01.npy", 
                  "zd_ann_prv_jg32_neocortex_z650-810_01.npy"]


    ann2_dsets = ["cj_ann_prv_jg05_neocortex_z310-449_01.npy", 
                  "cj_ann_prv_jg24_neocortex_z300-400_01.npy", 
                  "cj_ann_prv_jg29_neocortex_z300-500_01.npy", 
                  "cj_ann_prv_jg32_neocortex_z650-810_01.npy"]

    #get number of cells annotated by both users
    ann_cells = sum([len(v) for k,v, in points_dict.items() if k in ann1_dsets or k in ann2_dsets])
    print("\ncells annotated by 2 users: {}".format(ann_cells))
    
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
