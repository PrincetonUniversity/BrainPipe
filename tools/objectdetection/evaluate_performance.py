#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:04:27 2017

@author: tpisano
"""
import numpy as np, os, time, sys, collections, multiprocessing as mp, pandas as pd, gc, h5py, shutil
from tools.utils.directorydeterminer import directorydeterminer as dd, pth_update
from tools.utils.io import save_kwargs, makedir, load_np, load_dictionary, listdirfull, writer, make_memmap_from_np_list, load_memmap_arr, chunkit
from tools.utils.overlay import tile
from functools import partial, update_wrapper
from math import ceil
from skimage.external import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import load_model
import keras.backend as k
from tools.conv_net.functions.bipartite import pairwise_distance_metrics_multiple_cutoffs
from scipy.ndimage.morphology import generate_binary_structure
import cv2
from skimage.morphology import disk
from tools.utils.overlay import tile

if __name__ == '__main__':
    1
    #paired_centers(gt, pred, paired)
    #visualize_centers(inputs, labels=False)
    
#%%
def compute_pixel_loss(y,y_pred,thresh):
	y = y > 0.4
	y_pred = y_pred > thresh

	TP = 1.0*np.sum(np.logical_and(y,y_pred))
	TN = 1.0*np.sum(np.logical_and(~y,~y_pred))
	FP = 1.0*np.sum(np.logical_and(~y,y_pred))
	FN = 1.0*np.sum(np.logical_and(y,~y_pred))
    
    #value that gives some gradient of NN success. Smaller is better.
    #weight = np.sum(np.absolute(np.subtract(y,y_pred)))
	weight = np.sum(np.absolute(np.subtract((y+1)**10,(y_pred+1)**10))) #this penalizes missing values closer to one more

	if TP + FN == 0:
		tpr = 0.0
	else:
		tpr = TP/(TP + FN) #sens

	fpr = FP/(FP + TN) #spec
	ave = np.mean(y)
	if thresh == 0.0:
		print(ave)
	return tpr,fpr, ave, TP, TN, FP, FN, weight

def roc_curve(y,y_pred, dst=False):
    thresh_range = np.logspace(-4,0,200)
    fp_range = np.zeros_like(thresh_range)
    tp_range = np.zeros_like(thresh_range)
    for i,thresh in enumerate(thresh_range):
        tpr,fpr, ave, TP, TN, FP, FN, weight = compute_pixel_loss(y,y_pred,thresh)
        fp_range[i] = fpr
        tp_range[i] = tpr
    plt.figure()
    plt.plot(1-fp_range,tp_range)
    plt.title('ROC curve, varying threshold')
    plt.xlabel('1-FP')
    plt.ylabel('TP')
    plt.xlim([1.01, -0.01])
    plt.ylim([-0.01,1.01])
    plt.grid()
    if dst: plt.savefig(dst+'_roc_curve.pdf', dpi=300, transparent=True)
    plt.figure()
    plt.title('Ranges for TP(green) and FP(red)')
    plt.semilogx(thresh_range,fp_range,'r')
    plt.semilogx(thresh_range,tp_range,'g')
    plt.xlabel('Theshold')
    if dst: plt.savefig(dst+'tp_fp_ranges.pdf', dpi=300, transparent=True)
    plt.show()

    return

def compute_p_r_f1(tp, fn, fp):
    '''
    '''

    try:
        if isinstance(tp, collections.Iterable): tp = len(tp)
        if isinstance(fn, collections.Iterable): fn = len(fn)
        if isinstance(fp, collections.Iterable): fp = len(fp)
    except TypeError:
        pass

    try:
        #precision or PPV = tp/tp+fp
        p = tp / float(tp + fp)

        #recall or TPR= tp / tp + fn
        r = tp / float(tp + fn)

        #f1 = 2 * (p*r)/(p+r)
        f1 = 2 * float(p*r)/float(p+r)
        return p, r, f1
    except ZeroDivisionError:
        return 0,0,0


def compute_performance(paired, unpaired, unpaired_from, threshold):
    #classify true positives based on threshold distance
    tp = [xx for xx in paired if xx[2] <= threshold]

    fn = [xx[0] for xx in paired if xx[2] > threshold]
    if unpaired_from == 'groundtruth': fn = fn + unpaired

    fp = [xx[1] for xx in paired if xx[2] > threshold]
    if unpaired_from == 'prediction': fp = fp + unpaired

    return tp, fn, fp

def paired_centers(gt, pred, paired, dst=False):
    '''
    gt = kwargs['ground_truth']
    pred = kwargs['centers']
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xticks([])

    #get indices
    gtt = list(gt); predd = list(pred)
    g_del = [xx[0] for xx in paired]; g_del.sort(reverse=True)
    p_del = [xx[1] for xx in paired]; p_del.sort(reverse=True)

    for i in range(len(g_del)):
        del gtt[g_del[i]]
        del predd[p_del[i]]

    #False Negatives
    for i in gtt:
        z,y,x = i
        ax.scatter(z,y,x, c='k', marker='^')

    #False Positives
    for i in predd:
        z,y,x = i
        ax.scatter(z,y,x, c='k', marker='X')

    ax.legend(['False Negatives ({})'.format(len(gtt)), 'False Positives ({})'.format(len(predd))])

    for i, pair in enumerate(paired):
        c = np.random.rand(3,)
        zg, yg, xg = gt[pair[0]]
        zp, yp, xp = pred[pair[1]]
        ax.scatter(zg, yg, xg, c=c)
        ax.scatter(zp, yp, xp, c=c)
        ax.plot3D([zg, zp],[yg, yp],[xg, xp], c=c)

    #plt.title('True Positives ({})'.format(len(g_del)))
    
    if dst: plt.savefig(dst+'_paired_centers.pdf', dpi=300, transparent=True)

    return



def visualize_centers(inputs, labels=False, dst=False):
    '''
    inputs = [array([[ 47, 283, 198],[100, 142, 139],[101, 126, 234]])]
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = ['r','g','c','b', 'k', 'y']

    for i,cen in enumerate(inputs):
        z0,y0,x0 = np.asarray(cen).T
        ax.scatter(z0,y0,x0, c=color[i])

    if labels: ax.legend((labels))
    
    if dst: plt.savefig(dst+'_visualize_centers.pdf', dpi=300, transparent=True)

    return

def visualize_random_forests(rf, dst):
    '''
    dst = '/home/wanglab/wang/pisano/figures/cell_detection/randomforest/trees'
    '''
    from sklearn.tree import export_graphviz
    from sklearn import tree
    i_tree = 0
    for tree_in_forest in rf.estimators_:
        fl = os.path.join(dst,'tree_' + str(i_tree) + '.dot')
        with open(fl, 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        os.system('dot -Tpng {} -o {}'.format(fl, fl[:-3]+'png'))
        os.remove(fl)
        i_tree = i_tree + 1
    return

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, dst=False):
    """
    #from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from sklearn.metrics import confusion_matrix
    plt.figure(); np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    if dst:
        if dst[-4:] != '.pdf': dst = dst+'.pdf'
        plt.savefig(dst, dpi=300, transparent=True)
    return


    

    
