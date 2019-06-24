#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:35:51 2018

@author: tpisano
"""
## Find TP vs FP
import numpy as np, pandas as pd, os, sys, matplotlib.pyplot as plt, multiprocessing as mp, time
from tools.utils.io import listdirfull, load_np, makedir, save_kwargs,load_dictionary
from tools.objectdetection.evaluate_performance import pairwise_distance_metrics_multiple_cutoffs
from tools.objectdetection.postprocess_cnn import load_tiff_folder, probabilitymap_to_centers_thresh
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.externals import joblib
from skimage.external import tifffile


if __name__ == '__main__':
    from tools.objectdetection.train_random_forest import bin_data, train_random_forest, plot_roc, apply_random_forest

    #Parameters
    size = (7,30,30) #Size: of cube around each center to train on.
    cutoff = 40 #Cutoff: allowable distance (IN PIXELS) for a True positive
    n_estimator = 50 #number of trees
    max_depth = 25 #depth of forest
    cores = 10 #for parallelization
    kfold_splits = 5 # number of times to iterate through using kfold cross validation
    balance = False
    maxip=10
    pad = False #important for edge cases in training set (i.e. points that don't have sufficient border around them)
    dst = False #place to save classifier
    verbose = True
    warm_start = True
    precision_score = None#'micro' #'weighted'
        #'binary':Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
        #'micro':Calculate metrics globally by counting the total true positives, false negatives and false positives.
        #'macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        #'weighted':Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
        #'samples':Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    

    #parse data
    #cnn predicted output
    cnn_output = '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays_cnn_output'

    #input for cnn (raw data, ground_truth labels)
    training_data_src = '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays'

    #parse
    dct = bin_data(cnn_output, training_data_src, size=size, cutoff = cutoff, pad = pad, verbose=verbose, maxip=maxip)
    tps = dct['true_positives']; fps = dct['false_positives']; gts = dct['ground_truths']

    #train
    kwargs = train_random_forest(tps, fps, n_estimator = n_estimator, max_depth = max_depth, balance = balance, cores = cores, kfold_splits = kfold_splits, dst = dst, average=precision_score)

    #plot
    #%matplotlib inline
    save = '/home/wanglab/Downloads/rf'
    plot_roc(save=save, **kwargs)

    #apply
    cnn_src = listdirfull(cnn_output); cnn_src.sort(); cnn_src = cnn_src[0]
    inn = listdirfull(training_data_src); inn.sort();
    raw_src = load_np(inn[0])[0]
    gt = load_np(inn[0])[1]
    centers = apply_random_forest(kwargs['classifier'], raw_src, cnn_src, size = (7,25,25))

    #show
    from tools.conv_net.functions.dilation import dilate_with_element, ball
    gt = dilate_with_element(gt, ball(5))
    src = np.zeros_like(gt)
    for c in centers.astype('int'):
        src[c[0],c[1],c[2]] = 1
    src = dilate_with_element(src, ball(5))

    #Sweep: <-- usually performance is not affected that much by this
    for n_estimator in (10,20,50,100):
        for max_depth in (5,10,20,50,100):
            print('\n\n n_estimator--{}, max_depth--{}'.format(n_estimator, max_depth))
            train_random_forest(tps, fps, n_estimator = n_estimator, max_depth = max_depth, cores = cores, test_size = test_size, dst = dst)



def apply_random_forest(classifier, raw_src, cnn_src, collect_cnn = False, size = (3,12,12), pad=False, cores=4, numZSlicesPerSplit=300, overlapping_planes = 20, chunks=10, maxip=0):
    ''' THIS IS MEMORY INEFFICIENT - SEE random_forest.py for better functions
    classifier = pretrained random forest or path to pretrained random forest
    raw_src = folder of tiffs of raw input data
    cnn_src = folder of tiffs from cnn output
    pad = if True, pad the edges of objects determined. False: remove edge cases, usually better since they will be few in number
    cores = number of cores for parallelization, larger the number the less memory efficient
    numZSlicesPerSplit: chunk of zplanes to process at once. Adjust this and cores based on memory constraints.
    overlapping_planes: number of planes on each side to overlap by, this should be a comfortable amount larger than the maximum z distances of a single object
    chunks = number of chunks to divide connected components by. The larger the number the more memory efficiency, but a bit more IO required
    collect_cnn = optional to include cnn data for random forest input
    Returns a dictionary consisting of k=centers, v=corresponding pixel indices determine by CNN

    TO DO - MAKE SURE MAPPING STAYS THE SAME AND ORDER IS NOT LOST

    classifier = '/home/wanglab/wang/pisano/Python/lightsheet/supp_files/h129_rf_classifier.pkl'
    cnn_src = '/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays_cnn_output/20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1350-1700_x3100-3450'
    raw_src =  load_np('/home/wanglab/wang/pisano/conv_net/annotations/better_res/h129_memmap_arrays/20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1350-1700_x3100-3450.npy')[0]
    '''
    rf = joblib.load(classifier) if type(classifier) == str else classifier

    #load and find centers from cnn
    center_pixels_dct = probabilitymap_to_centers_thresh(cnn_src, threshold = 1, numZSlicesPerSplit=numZSlicesPerSplit, overlapping_planes = overlapping_planes, cores = cores, return_pixels = True, verbose = False)

    #optional
    if collect_cnn == False: cnn_src = False

    #load and collect pixels - if path to folder of tiffs will be done in memory efficient way
    if type(raw_src) == str and raw_src[:-4] == '.npy':  inn = get_pixels_around_center(pnts=center_pixels_dct.keys(), src=load_np(raw_src), cnn_src=cnn_src, size = size, pad=pad, return_pairs=True, maxip=maxip)
    elif type(raw_src) == str and raw_src[:-4] == '.tif': inn = get_pixels_around_center(pnts=center_pixels_dct.keys(), src=tifffile.imread(raw_src), cnn_src=cnn_src, size = size, pad=pad, return_pairs=True, maxip=maxip)
    elif type(raw_src) == str: inn = get_pixels_around_center_mem_eff(pnts=center_pixels_dct.keys(), src=raw_src, cnn_src=cnn_src, size = size, pad=pad, return_pairs=True, cores=cores, chunks=chunks,maxip=maxip)
    elif str(type(raw_src)) in ["<class 'numpy.core.memmap.memmap'>", "<type 'numpy.ndarray'>"]: inn = get_pixels_around_center(pnts=center_pixels_dct.keys(), src=raw_src, cnn_src=cnn_src, size = size, pad=pad, return_pairs=True, maxip=maxip)

    #predict
    out = rf.predict(inn.values())

    #remove centers that are considered false positives
    centers = np.asarray([xx for i,xx in enumerate(inn.keys()) if out[i]==1])

    #remove non determine centers from above
    center_pixels_dct = {tuple(c):center_pixels_dct[tuple(c)] for c in centers}

    return center_pixels_dct


def bin_data(cnn_output, training_data_src, size = (3, 12, 12), cutoff = 15, pad = True, verbose = False, collect_cnn = False, maxip=0):
    '''collect connected components from CNN and bin into True positives and false positives based on labels

    cnn_output = pth/list of folders containing CNN output
    training_data_src = pth/list of folders containing numpy arrays [c,z,y,x]
                    c = 0: raw data, c=1: nonzeropixels representing ground truth centers, requiring same naming as src_cnn
    size = used in get_pixels_around_center; distance from point in zyx.
        Note: this is effectively a radius (NOT DIAMETER).
        note:
    cutoff = allowable distance (IN PIXELS) for a True positive when considering ground truth centers with centers of mass of cnn-labelled connected components
    pad = (optional) important for edge cases in training set (i.e. points that don't have sufficient border around them)
        True if pnt is on edge of image, function pads evenly
        Flase if pnt is on edge of image, drop
    collect_cnn (optional): if true keep these data for training as well
    maxip = int, number of maxips to ravel into data
    return:
        {'true_positives': tps, 'false_positives': fps, 'ground_truths': gts}

    '''
    cnn_output = listdirfull(cnn_output) if type(cnn_output) == str else cnn_output
    training_data_src = listdirfull(training_data_src) if type(training_data_src) == str else training_data_src
    intersection = list(set([os.path.basename(xx[:-4]) for xx in training_data_src]).intersection(set([os.path.basename(xx) for xx in cnn_output])))
    if verbose: print('Collect cnn == {}'.format(collect_cnn))
    tps=[]; fps=[]; gts = []
    for pth in intersection:
        if verbose: sys.stdout.write('Starting {}'.format(pth))

        #load raw and gts
        data = load_np(os.path.join(os.path.dirname(training_data_src[0]), pth+'.npy'))
        raw = data[0]
        ground_truth = data[1]
        gt = np.asarray(np.nonzero(ground_truth)).T

        #get labels and pair based on distance
        centers = probabilitymap_to_centers_thresh(os.path.join(os.path.dirname(cnn_output[0]), pth), threshold = 1, numZSlicesPerSplit=250, overlapping_planes = 40, cores = 4, verbose = verbose)
        try:
            paired, tp, fp, fn = pairwise_distance_metrics_multiple_cutoffs(gt, centers, verbose=False, return_paired=True, cutoffs=[cutoff])[0]
    
            #optional
            cnn_src = os.path.join(os.path.dirname(cnn_output[0]), pth) if collect_cnn == True else False
            TP = [centers[xx[1]] for xx in paired]
            TPS = get_pixels_around_center(np.asarray(TP).astype('int'), raw, cnn_src=cnn_src, size=size, pad=pad, maxip=maxip)
            FP = np.asarray(list(set(centers).difference(set(TP))))
            FPS = get_pixels_around_center(np.asarray(FP).astype('int'), raw, cnn_src=cnn_src, size=size, pad=pad, maxip=maxip)
            
            #append
            tps.append(TPS); fps.append(FPS); gts.append(gt)
        except Exception, e:
            break
            print ('\n\n\nSkipping {}, due to error: {}\n\n\n'.format(pth, e))
            
    #clean
    tps = [xx for xx in tps if xx.ndim>1]
    fps = [xx for xx in fps if xx.ndim>1]
    gts = [xx for xx in gts if xx.ndim>1]

    #parse into numpy arrays:
    tps = np.concatenate(tps, 0)
    fps = np.concatenate(fps, 0)
    gts = np.concatenate(gts, 0)

    #determine number of gts dropped
    if verbose: print('{} of {} Ground truth positives dropped, because they are on the border of an image. To keep more decrease size of in bin_data'.format(len(gts) - len(tps), len(gts)))

    return {'true_positives': tps, 'false_positives': fps, 'ground_truths': gts}


def train_random_forest(tps, fps, n_estimator = 20, max_depth = 10, cores = 10, kfold_splits = 10, class_weight = None, dst = False, balance=False, max_features=None, precision_score='micro', warm_start=True, test_size=0.1):
    '''
    Function to train sklearn's RandomForestClassifier
    Info:
        #http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

    To load and predict:
        rf = joblib.load(pth)
        rf.predict(X_test[0].reshape(1,-1)) <-1,m
        rf.predict(X_test) <-n,m

    Inputs
    ------------
    tps = n,m numpy array of true positivies generated from bin_data
    fps = n,m numpy array of false positivies generated from bin_data
    kfold_splits = number of times to cross validate with sklearn's kfold. Trains on 90%, tests on 10%
    balance = TRAINING ONLY
        False include all data in training
        True make sure the number of training examples for Cell and not Cell are the same. Assumption is more non cells than cells
    class_weight = optional see sklearn.ensemble.randomforest's class_weight
    max_features= optional see sklearn.ensemble.randomforest
    '''
    #verbose = 1 if verbose else 0

    #inputs
    X = np.concatenate((tps,fps), axis=0)
    y = np.asarray([1 for xx in tps] + [0 for xx in fps])
    
    #init
    best_accuracy=0; best_rf = 0; accuracy_lst=[]

    #KFold cross validation - using straified kfold as it preserves ratios between classes
    if kfold_splits>1:
        gen = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=np.random.seed(10))
    else:
        from sklearn.model_selection import ShuffleSplit
        gen = ShuffleSplit(n_splits = 1, test_size=test_size, random_state=np.random.seed(10))
    for train_index, test_index in gen.split(X, y):
        st = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #optional balance
        if balance:
            ln = len(np.where(y_train == 1)[0])
            idxs = np.concatenate((np.where(y_train == 0)[0][:ln],np.where(y_train == 1)[0][:ln]), axis=0)
            X_train = X_train[idxs]; y_train = y_train[idxs]

        #Fit supervised transformation based on random forests
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.metrics import f1_score
        rf = ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimator, n_jobs = cores, class_weight=class_weight, max_features=max_features, warm_start=warm_start)
        rf.fit(X_train, y_train)


        #Performance
        accuracy = rf.score(X_test, y_test); accuracy_lst.append(accuracy)
        f1 = f1_score(y_test, rf.predict(X_test), average=precision_score)
        print ('Classifier accuracy: {}, f1: {} in {} min'.format(accuracy, f1, round((time.time() - st)/60, 2)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_rf = rf
            x_train_to_use_for_roc = X_train
            x_test_to_use_for_roc = X_test
            y_train_to_use_for_roc = y_train
            y_test_to_use_for_roc = y_test


    out = {'true_positives': tps, 'false_positives': fps, 'X_train': x_train_to_use_for_roc, 'X_test': x_test_to_use_for_roc,
           'y_train': y_train_to_use_for_roc, 'y_test': y_test_to_use_for_roc, 'classifier': best_rf, 'accuracy': np.mean(accuracy_lst)}

    #save out
    if dst:
        if dst[-4:] != '.pkl': dst = dst+'.pkl'
        joblib.dump(best_rf, dst)
        out['dst'] = dst

    return out


def plot_roc(save=False, **kwargs):
    '''From http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

    save (optional place to save)
    kwargs = from train_random_forest

    '''
    from sklearn.metrics import roc_curve
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']
    rf = kwargs['classifier']

    # The random forest model by itself
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve, classifier accuracy: ({})'.format(np.round(kwargs['accuracy'],decimals=4)))
    plt.legend(loc='best')
    plt.show()
    if save: plt.savefig(save+'_roc.pdf', dpi = 300)

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()

    if save: plt.savefig(save+'_roc_zoomed_in.pdf', dpi = 300)
    return


def get_pixels_around_center_mem_eff(pnts, src, cnn_src = False, size=(7,50,50), pad = False, return_pairs = True, chunks = 10, cores=1, maxip=0):
    '''Wrapper function for get_pixels_around_center to be memory efficient

    src needs to be a folder of tiffs here
    '''
    from tools.utils.io import chunkit

    #sort points in z:
    pnts = sorted(pnts, key=lambda a: a[0])

    #chunk
    iterlst = [(pnts[chunkit(chunk, chunks, pnts)[0]: chunkit(chunk, chunks, pnts)[1]], src, cnn_src, size, pad, return_pairs, maxip) for chunk in range(chunks)]

    #run
    if cores <=1:
        out = [get_pixels_around_center_mem_eff_helper((i)) for i in iterlst]
    else:
        p = mp.Pool(cores)
        out = p.map(get_pixels_around_center_mem_eff_helper, iterlst)
        p.terminate()

    #unpack
    if return_pairs:
        out = {k:v for d in out for k,v in d.iteritems()}
    elif not return_pairs:
        out = np.concatenate(out, 0)

    return out

def get_pixels_around_center_mem_eff_helper((pnts, src, cnn_src, size, pad, return_pairs, maxip)):
    '''Helper. Assumes src path to list of tiffs.

    Loads appropriate tiff range, adjust z planes, grabs pixels, and then readjust zplanes

     src_range = (409, 491)
    '''
    #find min and max of presorted sub points and account for window size
    zrng = tuple((int(pnts[0][0]) - size[0], int(pnts[-1][0]) + 1 + size[0]))

    #load that chunk
    src = load_tiff_folder(src, threshold = 0, load_range = (zrng[0], zrng[1]))
    if cnn_src: cnn_src = load_tiff_folder(cnn_src, threshold = 1, load_range = (zrng[0], zrng[1]))

    #subtract from points to account for volume chunk
    pnts = [tuple((xx[0] - zrng[0], xx[1], xx[2])) for xx in pnts]

    #get pixels
    out = get_pixels_around_center(pnts, src, cnn_src, size=size, pad = pad, return_pairs = return_pairs, maxip=maxip)

    #readjust for pixel shifts, don't need to do if return_pairs is false, since it is an n,m array (n=num points, m = size.ravel())
    if return_pairs: out = {tuple((k[0]+zrng[0], k[1], k[2])):v for k,v in out.iteritems()}

    return out

def get_pixels_around_center(pnts, src, cnn_src = False, size=(3,12,12), pad = False, return_pairs = False, maxip=None):
    '''Function to return np.ravel() pixels of a given cube dims of centers

    pnts = zyx points for surrounding pixels to return
    src = volume representing raw data
    size = distance from point in zyx. Note this is effectively a radius (NOT DIAMETER).
    pad = (optional) important for edge cases in training set (i.e. points that don't have sufficient border around them)
        True if pnt is on edge of image, function pads evenly
        Flase if pnt is on edge of image, drop
    return_pairs = maps pnts with out
    cnn_src THIS IS JUST A SECOND COPY OF RAW DATA _ WORKS BETTER AND INTENTIONAL, one is normalized other is not
    maxip = int, number of maxips to ravel into data

    '''
    #inputs
    zr,yr,xr = size
    dims = (2*zr + 1,2*yr+1, 2*xr+1)
    if type(cnn_src) == str: cnn_src = load_tiff_folder(cnn_src, threshold=1)
    #ravel length, if cnn_src should be doubled
    ln = 2*np.zeros(dims).ravel().shape[0] if str(type(cnn_src)) == "<type 'numpy.ndarray'>" else np.zeros(dims).ravel().shape[0]
    if maxip>0: ln = ln+(np.zeros(dims[1:]).ravel().shape[0]*maxip)

    #iterate through points
    output = {}
    for pnt in pnts:
        #src
        z,y,x=[xx.astype('int') for xx in pnt]
        data = adjust_boundary(np.copy(src[z-zr:z+zr+1,y-yr:y+yr+1,x-xr:x+xr+1]), dims) if pad else np.copy(np.copy(src[z-zr:z+zr+1,y-yr:y+yr+1,x-xr:x+xr+1]))
        
        #out
        out = norm(np.copy(data).ravel())
        
        #cnn_src - this is basically a non normalized copy of the raw
        if str(type(cnn_src)) == "<type 'numpy.ndarray'>":
            cnn_out = data.ravel()
            out = np.concatenate((out, cnn_out),axis=0)

        #maxip, CONSIDER NORMALIZING
        if maxip>0 and data.shape[0]>0:
            maxx = np.max(data, 0).ravel()
            for i in range(maxip):
                out = np.append(out, maxx)
        
        #add to dct
        if out.shape[0] == ln:
            output[tuple(pnt)] = out

    if not return_pairs: return np.asarray(output.values())
    if return_pairs: return output

def norm(vol):
    try:
        vol = (vol - vol.min()) / (vol.max() - vol.min())
    except:
        vol = vol
    return vol


def adjust_boundary(src, dims):
    '''Adjust boundary of a numpy array to fit provided dimensions. Pads evenly

    Can only remove or add, not a combination of both

    src = np.zeros((224,224))
    dims = (212,212)

    '''
    assert len(src.shape) == len(dims), 'Need to provide same number of dimensions for src and dims'

    #if equal
    if src.shape == dims: return src

    delta = [int((a-b)/2.0) for a,b in zip(src.shape, dims)]

    assert np.any((np.all([xx<=0 for xx in delta]), np.all([xx>=0 for xx in delta]))) #ensures either expanding or removing, not both

    if delta[0]>=0:
        if len(src.shape) == 2: src = eval('src[{0}:-{0},{1}:-{1}]'.format(*delta))
        if len(src.shape) == 3: src = eval('src[{0}:-{0},{1}:-{1}, {2}:-{2}]'.format(*delta))
    elif delta[0]<0:
        delta = [int(abs(xx)) for xx in delta]
        if len(src.shape) == 2: src = eval('np.pad(src, mode="constant", pad_width=(({0},{0}), ({1},{1})))'.format(*delta))
        if len(src.shape) == 3: src = eval('np.pad(src, mode="constant", pad_width=(({0},{0}), ({1},{1}), ({2},{2})))'.format(*delta))

    if src.shape != dims:
        delta = [abs(a-b) for a,b in zip(src.shape, dims)]
        if len(src.shape) == 2: src = eval('np.pad(src, mode="constant", pad_width=(({0},0), ({1},0)))'.format(*delta))
        if len(src.shape) == 3: src = eval('np.pad(src, mode="constant", pad_width=(({0},0), ({1},0), ({2},0)))'.format(*delta))

    return src

def remove_border(src, size):
    '''Function to remove border from evaluation sets. This is important because the forest needs the size border around each ground truth
    '''
    z,y,x = size
    return src[z:-z, y:-y, x:-x]

def zero_border(src, size):
    '''Function to zero border from evaluation sets. This is important because the forest needs the size border around each ground truth
    '''
    nsrc = np.zeros_like(src)
    z,y,x = size
    nsrc[z:-z, y:-y, x:-x] = src[z:-z, y:-y, x:-x]
    return nsrc

def randomsearch():
    '''Container to put code
    '''
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    from sklearn.model_selection import RandomizedSearchCV
    #inputs
    X = np.concatenate((tps,fps), axis=0)
    y = np.asarray([1 for xx in tps] + [0 for xx in fps])
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2000, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [50,80,120]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    
    rf = ExtraTreesClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=10)
    rf_random.fit(X, y)
    rf_random.best_params_
    
    {'bootstrap': False,
     'max_depth': 80,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'n_estimators': 822}
    
    
    {'bootstrap': False,
     'max_depth': 50,
     'max_features': 'auto',
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'n_estimators': 800}

    return