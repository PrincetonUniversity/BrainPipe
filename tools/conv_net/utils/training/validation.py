# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: zahramansoor
"""

import matplotlib.pyplot as plt, os
import h5py, numpy as np
from scipy.stats import linregress

def save_stats_h5(fname):
    '''Function to extract test loss and training loss values from h5 files saved in training.
    '''

    with h5py.File(fname) as f:
        print('keys of file:\n {}'.format(list(f.keys())))
        print('base lr value: {}'.format(f['base_lr'].value))
        test = list(f['test'].keys())
        print('contents of test dict: \n {}'.format(test))
        train = list(f['train'].keys())
        print('contents of train dict: \n {}'.format(train))
        test_loss_arr = f['test'][test[2]].value
        train_loss_arr = f['train'][train[2]].value
        
    return test_loss_arr, train_loss_arr


def plot_val_curve(loss, start_iter = 0, end_iter = 15000, m = 10):
    '''Function to plot validation data loss value from h5 file from training on tiger2
    Inputs:
        loss = array of loss values
        pth = path to save pdf
        fname = file name (as string) to add onto pdf
        start_iter = iteration from which to start plotting from, default is 0
        m = multiple at which log was saved (in parameter dictionary), default is 10
    '''
    #set x and y
    iters = np.arange(0, len(loss[:end_iter]))
    if len(loss) > 1000: 
        loss = np.take(loss[:end_iter], np.arange(0, len(loss[:end_iter])-1, m)) 
        iters = np.take(iters[:end_iter], np.arange(0, len(iters)-1, m))
    
    #linear regression
    fit = np.polyfit(iters[start_iter:end_iter], loss[start_iter:end_iter], 1)
    fit_fn = np.poly1d(fit)
    linreg_stats = linregress(iters[start_iter:end_iter], loss[start_iter:end_iter])
    loss
    #plot
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    plt.plot(loss[start_iter:end_iter], 'ro')
    plt.xlabel('# of iterations in thousands')
    plt.ylabel('loss value')
    plt.title('3D U-net validation curve for C-FOS')          
    plt.show()
    
    plt.figure()
    plt.plot(loss[start_iter:end_iter], 'yo', fit_fn(iters[start_iter:end_iter]), '--k')
    plt.xlabel('# of iterations in thousands')
    plt.ylabel('loss value')
    plt.title('Linear regression of loss values')  
    plt.show()
    
    return linreg_stats