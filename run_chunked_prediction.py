#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:41:28 2018

@author: wanglab
"""
from neurotorch.datasets.dataset import Array
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.core.predictor import Predictor
import torch
import numpy as np, sys, time, os


def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.
    
    by @tpisano

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr


def run_prediction(data_pth, chkpnt_num, verbose = False):    
    '''
    Main function to run neurotorch prediction functions using patched large data set.
    
    Inputs:
        data_pth = directory in which patched input memory mapped array is stored 
        chkpnt_num = checkpoint from training to run prediction
    Returns:
        path of patched probability array needed for reconstruction
    '''
    sys.stdout.write('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct - use 0.4.1
    
    net = torch.nn.DataParallel(RSUNet())  #initialize the U-Net architecture - use torch.nn.DataParallel if you used this to train the net using nick turner's pytorchutils
    
    inputs = load_memmap_arr(os.path.join(data_pth, 'patched_memmap_array.npy')) #load input patched array 
    out_map = load_memmap_arr(os.path.join(data_pth, 'patched_prediction_array.npy'), mode = 'w+', shape = inputs.shape) #initialise output probability map
    
    predictor = Predictor(net, checkpoint = chkpnt_num, gpu_device = 0) #setup a predictor for computing outputs
    
    initial = time.time()
    
    for i in range(len(inputs[:,0,0,0])): #iterates through each large patch to run inference #len(inputs[0])       
        
        start = time.time()
        inpt_dataset = Array(inputs[i,:,:,:]) #grab chunk
        
        if verbose:
            sys.stdout.write('*******************************************************************************\n\
           Starting predictions for patch #: {} of {} \n\n'.format(i, len(inputs[:,0,0,0]))); sys.stdout.flush()
        
        out_dataset = Array(out_map[i,:,:,:]) #initialise output array of chunk
        predictor.run(inpt_dataset, out_dataset, batch_size = 22)  #run prediction

        if verbose: sys.stdout.write('Finishing predictions & saving :]... '); sys.stdout.flush() 
        out_map[i,:,:,:] = out_dataset.getArray().astype(np.float32) #save output array into initialised probability map
        if i%25==0: out_map.flush()
        
        sys.stdout.write('Elapsed time: {} minutes\n'.format(round((time.time()-start)/60, 1))); sys.stdout.flush()
        
    sys.stdout.write('Time spent predicting: {} minutes'.format(round((time.time()-initial)/60, 1))); sys.stdout.flush()
    
    return os.path.join(data_pth, 'patched_prediction_array.npy')

#%%    
if __name__ == '__main__':  
    
    data_pth = '/home/wanglab/Documents/data/20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00'
    chkpnt_num = '/jukebox/wang/zahra/conv_net/training/experiment_dirs/20181009_zd_train/models/model995000.chkpt'
    
    run_prediction(data_pth, chkpnt_num)
    