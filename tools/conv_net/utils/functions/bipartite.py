# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:06:23 2017

@author: tjp7rr1
"""

import numpy as np, sys, matplotlib.pyplot as plt
from scipy.spatial import distance
import time

def pairwise_distance_metrics_given_cdists(ground_truth, predicted, y, cutoff=10, verbose=True):
    """Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
    y: matrix of distances between all elements of ground truth and predicted
                        
    Returns:
    -------
    paired: list of [ground_truth"s index (from input list), predicted"s index (from input list), distance]
    tp,fp,fn: statistics on true positives, false positives, and false negatives.
    """   
    
    start=time.time()
    if verbose: print("\nCalculating pairwise distances at cutoff {}...".format(cutoff))
    st = time.time()
    #only keep those distances that are below the cutoff!
    truth_indices,pred_indices = np.where(y <= cutoff)
    dists = zip(y[truth_indices,pred_indices],truth_indices,pred_indices)
    if verbose: print("  {} seconds for calculating and collecting distances".format(np.round(time.time() - st, decimals=3)))

    #sort by smallest dist
    if verbose: print("Sorting..."); st = time.time()
    dists=sorted(dists, key=lambda x: x[0])
    if verbose: sys.stdout.write("  Sorted in {} seconds.".format(np.round(time.time() - st, decimals=3))); sys.stdout.flush()

    used_truth = set()
    used_pred = set()
    paired = []
    for (i,dist) in enumerate(dists):
        d = dist[0]
        if d > cutoff:
            #we have reached distances beyond the cutoff
            if verbose: print("Reached cutoff distance, so far {} paired".format(len(paired)))
            break
        truth_idx = dist[1]
        pred_idx = dist[2]
        if truth_idx not in used_truth and pred_idx not in used_pred:
            paired.append((truth_idx,pred_idx,d))
            used_truth.add(truth_idx)
            used_pred.add(pred_idx)
        if len(used_truth) == len(ground_truth) or len(used_pred) == len(predicted):
            # we have used up all the entries from the shorter list
            break

    tp = len(paired)
    fn = len(ground_truth) - len(paired)
    fp = len(predicted) - len(paired)
    if verbose: print("TP: {}, FP: {}, FN: {}".format(tp,fp,fn))
    # print(paired)
  
    if verbose: 
        plt.hist([xx[2] for xx in paired] , bins = np.max((int(len(paired)/500), 10)))
        plt.title("Histogram of distances - pixel or microns")
    
    if verbose: print("Finished in {} seconds\n".format(np.round(time.time() - start,decimals = 3)))
    
    return paired,tp,fp,fn


def pairwise_distance_metrics(ground_truth, predicted, cutoff=10, verbose=True):
    """Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
                        
    Returns:
    -------
    paired: list of [ground_truth"s index (from input list), predicted"s index (from input list), distance]
    tp,fp,fn: statistics on true positives, false positives, and false negatives.
    """   
    if verbose: print("\nCalculating pairwise distances...")
    y = distance.cdist(ground_truth, predicted, metric="euclidean")
    return pairwise_distance_metrics_given_cdists(ground_truth,predicted,y,cutoff,verbose)


def pairwise_distance_metrics_multiple_cutoffs(ground_truth, predicted, cutoffs=[0.1,1.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125], verbose=True, return_paired=False):
    """Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
                        
    Returns:
    -------
    paired: list of [ground_truth"s index (from input list), predicted"s index (from input list), distance]
    tp,fp,fn: statistics on true positives, false positives, and false negatives.
    return_paired (optional): returns paired, useful for visualizign
    """
    #input
    ground_truth = np.asarray(ground_truth)
    predicted = np.asarray(predicted)
    #deal with either gt and/or p being empty...
    if ground_truth.shape[0]==0 & predicted.shape[0]==0:
        tp, fp, fn = 0, 0, 0
        return [[tp, fp, fn] for i in range(len(cutoffs))]
    elif ground_truth.shape[0]==0:
        tp, fp, fn = 0, predicted.shape[0], 0
        return [[tp, fp, fn] for i in range(len(cutoffs))]
    elif predicted.shape[0]==0:
        tp, fp, fn = 0, 0, ground_truth.shape[0]
        return [[tp, fp, fn] for i in range(len(cutoffs))]
    #or do real calcs
    else:
        if verbose: print("\nCalculating pairwise distances...")
        y = distance.cdist(ground_truth, predicted, metric="euclidean")
        all_stats = [] #list of (tp,fp,fn) values
        for cutoff in cutoffs:
            if not return_paired: all_stats.append(pairwise_distance_metrics_given_cdists(ground_truth,predicted,y,cutoff,verbose)[1:]) #append only tp,fp, fn values
            if return_paired: all_stats.append(pairwise_distance_metrics_given_cdists(ground_truth,predicted,y,cutoff,verbose)[:]) #append only tp,fp, fn values
        return all_stats
  
            
def pairwise_distance(ground_truth, predicted, verbose=True):
    """Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
                        
    
    Returns:
    -------
    paired: list of [ground_truth"s coordiante, predicted"s cooridnate, distance]
    unpaired: list of [smaller of two(ground_truth, predicted)"s unpaired coordinates]
    unpaired_from: list which was larger and thus has unpaired...needed for determining if FP or FN
            "ground_truth", "predicted", "equal"
    
    
    #COULD IMPROVE PERFORMANCE USING AN ORDERED DICT INSTEAD OF DISTS LIST
    testing:

    """   
    
    start=time.time()
        
    #faster
    if verbose: print("\nCalculating pairwise distances...")
    st = time.time()
    y = distance.cdist(ground_truth, predicted, metric="euclidean")
    #format:
    dists = [[y[i,ii], ground_truth[i], predicted[ii]] for i in range(len(ground_truth)) for ii in range(len(predicted))]
    if verbose: print("  {} seconds for cdist\n".format(np.round(time.time() - st, decimals=3)))
    

    #sort by smallest dist
    if verbose: print("  Sorting..."); st = time.time()
    dists=sorted(dists, key=lambda x: x[0])
    if verbose: sys.stdout.write("{} seconds".format(np.round(time.time() - st, decimals=3))); sys.stdout.flush()


    #make dict of dict, faster than list iteration
    if verbose: sys.stdout.write("\n  Populating dictionary and pruning based on shortest distances..."); sys.stdout.flush()
    used={}; paired=[]; tick=0; i = 0
    while len(paired) < min((len(ground_truth),len(predicted))):    
        dist, aa, bb = dists[i]
        i+=1
        if str(aa) not in used and str(bb) not in used:
            paired.append([str(aa), str(bb), dist]); tick+=1
            used[str(aa)]=1; used[str(bb)]=1
            if verbose and tick%250 == 0: sys.stdout.write("\n   paired {} of {} total in {}sec".format(tick, min(len(ground_truth), len(predicted)), np.round(time.time() - start)));sys.stdout.flush()
        if verbose and i%1000000 == 0: sys.stdout.write("\n       iteration {} of max iteration {} in {}sec".format(i, len(dists), np.round(time.time() - start)));sys.stdout.flush()

            
    #find unpaired points:
    unpaired = set([str(xx) for xx in smaller_list(ground_truth, predicted)]).difference(set([xx[1] for xx in paired]))
    
    if verbose: 
        plt.hist([xx[2] for xx in paired] , bins = np.max((int(len(paired)/500), 10)))
        plt.title("Histogram of distances - pixel or microns")
    
    sys.stdout.write("\n\n  Pairs found {}. Unpaired {}. Unpaired come from {}".format(len(paired), len(unpaired), which_is_larger(ground_truth,predicted))); sys.stdout.flush()      
    
    #reformat to array
    paired = [[np.asarray([float(yy) for yy in xx[0].replace("(","").replace(")","").replace("[","").replace("]","").replace(",","").split()]), np.asarray([float(zz) for zz in xx[1].replace("(","").replace(")","").replace(",","").split()]),xx[2]] for xx in paired]
    unpaired = [np.asarray([float(yy) for yy in xx.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","").split()]) for xx in unpaired]
    
    return paired, list(unpaired), which_is_larger(ground_truth,predicted)

def smaller_list(a,b):
    if len(a)<=len(b):
        return a
    else:
        return b
    
def which_is_larger(ground_truth,predicted):
    if len(ground_truth)>len(predicted):
        return "ground_truth"
    elif len(ground_truth)<len(predicted):
        return "predicted"
    else:
        return "equal"

def to_int(arr):
    return int("".join([str(i).zfill(6) for i in arr]))

if __name__ == "__main__":

    #dummy data
    size = 2000
    cutoff = 0.035
    ground_truth = np.asarray([np.random.rand(3) for xx in range(size+10)])
    predicted = np.asarray([xx + 0.01*np.random.randn(3) for xx in ground_truth[:size]])
    #%timeit pairwise_distance(detected, ground_truth, verbose=True)
        
    #
    paired, tp,fp,fn  = pairwise_distance_metrics(ground_truth, predicted, cutoff=cutoff,verbose=True)
    # print(paired)

 
