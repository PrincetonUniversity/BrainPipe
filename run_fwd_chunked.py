#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:41:28 2018

@author: wanglab
"""
from neurotorch.datasets.filetypes import TiffVolume
from neurotorch.datasets.dataset import PooledVolume, Array
from neurotorch.datasets.datatypes import BoundingBox, Vector
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.core.predictor import Predictor
import torch
from skimage.external import tifffile
import numpy as np, sys


def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.

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

#%%
def main():
    
    sys.stdout.write('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct - use 0.4.1
    
    net = torch.nn.DataParallel(RSUNet())  #initialize the U-Net architecture - use torch.nn.DataParallel if you used this to train the net
    
    data_pth = '/home/wanglab/Documents/data/chunk_test/patched_memmap_array.npy'
    
    inputs = load_memmap_arr(data_pth) #load input patched array
    
    out_pth = '/home/wanglab/Documents/data/chunk_test/probability_array.npy'
    
    out_map = load_memmap_arr(out_pth, mode = 'w+', shape = inputs.shape) #initialise output array
        
    for i in range(len(inputs[:,0,0,0])): #len(inputs[0])
        
        inpt_dataset = Array(inputs[i,:,:,:]) #grab chunk
        
        sys.stdout.write('*******************************************************************************\n\n\
           Starting predictions for patch #: {}...\n\n'.format(i))
    
        predictor = Predictor(net, checkpoint='/jukebox/wang/zahra/conv_net/training/20181009_zd_train/models/model995000.chkpt', 
                          gpu_device=0) #setup a predictor for computing outputs
        
        out_dataset = Array(out_map[i,:,:,:]) #initialise output array of chunk
        
        predictor.run(inpt_dataset, out_dataset, batch_size=2)  #run prediction

        sys.stdout.write('*******************************************************************************\n\n\
               Finishing predictions :) Saving... \n\n') 
    
        out_map[i,:,:,:] = out_dataset.getArray().astype(np.float32) #save output array into initialised probability map
        
        sys.stdout.write('*******************************************************************************\n\n\
                Saved!')

#%%    
if __name__ == '__main__':
    
    main()