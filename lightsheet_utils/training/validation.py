# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: Zahra
"""

import re, os
import matplotlib.pyplot as plt
import h5py, numpy as np

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



def plot_val_curve(pth, exclude_initial = False):
    '''Function to plot validation data loss value from .out file from training on tiger2
    Inputs:
        pth = directory of .out files from training
    Returns:
        pdf of plot, in the path directory
    '''
    
    #initialise things
    test = ''
    
    out = os.listdir(pth)
    pths = [os.path.join(pth, f) for f in out if f[-4:] == '.out']; pths.sort() #sort so trainings are in order of time performed
    
    #read text file output from PyTorchUtils
    for src in pths:
        with open(src, 'r') as searchfile:
            for line in searchfile:
                if 'TEST:' in line: #finds all lines with test
                    test += line
            searchfile.close()

    #regex    
    n = re.compile("(?<='soma_label':\s)(\d+.\d+)") #finds loss values in test lines
    loss = n.findall(test)
    loss = [round(float(xx), 5) for xx in loss if str(xx)] #write into numerical vector
    
    #plot
    if exclude_initial:
        plt.rcParams.update({'font.size': 8})
        plt.figure()
        plt.plot(loss[1:], 'r')
        plt.ylim(0, 0.0015)
        plt.xlabel('# of iterations in thousands')
        plt.ylabel('loss value')
        plt.title('3D U-net validation curve for H129')          
        plt.savefig(os.path.join(pth, 'val_zoom_initial_test_omitted.pdf'), dpi = 300)
        plt.close()
    else:
        plt.rcParams.update({'font.size': 8})
        plt.figure()
        plt.plot(loss, 'r')
        plt.ylim(0, 0.02)
        plt.xlabel('# of iterations in thousands')
        plt.ylabel('loss value')
        plt.title('3D U-net validation curve for H129')          
        plt.savefig(os.path.join(pth, 'val_zoom'), dpi = 300)
        plt.close()    