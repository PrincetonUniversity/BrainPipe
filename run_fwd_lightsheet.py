#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:19:15 2018

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

#%%
def main():
    
    sys.stdout.write('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct
    
    net = torch.nn.DataParallel(RSUNet())  #initialize the U-Net architecture - use torch.nn.DataParallel if you used this to train the net
    
    data_pth = '/home/wanglab/Documents/python/NeuroTorch/data'
    inputs_dataset = TiffVolume(data_pth, BoundingBox(Vector(0, 0, 0), Vector(3200, 3200, 20)))
    inputs_dataset.__enter__()
    
    sys.stdout.write('*******************************************************************************\n\n\
           Starting predictions...\n\n') 
    
    predictor = Predictor(net, checkpoint='/jukebox/wang/zahra/conv_net/training/20181009_zd_train/models/model715000.chkpt', 
                          gpu_device=0) #setup a predictor for computing outputs
    
    outputs = Array(np.zeros(inputs_dataset.getBoundingBox().getNumpyDim())) #initialise output array
    
    predictor.run(inputs_dataset, outputs, batch_size = 6)  #run prediction

    sys.stdout.write('*******************************************************************************\n\n\
           Finishing predictions :) Saving... \n\n') 
    
    tifffile.imsave("test_prediction.tif", outputs.getArray().astype(np.float32)) #saves image output, zeros = cells??? so bizarre
    
    sys.stdout.write('*******************************************************************************\n\n\
            Saved!')
    
if __name__ == '__main__':
    
    main()

#**********************************************************************************************************************************
#**********************************************************************************************************************************    
#example from package
#**********************************************************************************************************************************
#**********************************************************************************************************************************    
#if not os.path.isdir('./tests/checkpoints'):
#    os.mkdir('tests/checkpoints')
#
#net = RSUNet()
#
#checkpoint = './tests/checkpoints/iteration_10.ckpt'
#inputs_dataset = TiffVolume(os.path.join(IMAGE_PATH,
#                                         "sample_volume.tif"),
#                            BoundingBox(Vector(0, 0, 0),
#                                        Vector(1024, 512, 50)))
#inputs_dataset.__enter__()
#predictor = Predictor(net, checkpoint, gpu_device=1)
#
#output_volume = Array(np.zeros(inputs_dataset
#                                .getBoundingBox()
#                                .getNumpyDim()))
#
#predictor.run(inputs_dataset, output_volume, batch_size=5)
#
#tif.imsave(os.path.join(IMAGE_PATH,
#                        "test_prediction.tif"),
#           output_volume.getArray().astype(np.float32))