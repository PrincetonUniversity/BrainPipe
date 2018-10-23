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
import numpy as np, os, sys

#%%
def main():
    
    sys.stdout.write('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct
    
    net = torch.nn.DataParallel(RSUNet())  # Initialize the U-Net architecture
    
    data_pth = '/home/wanglab/Documents/python/NeuroTorch/data'
    inputs_dataset = TiffVolume(data_pth, BoundingBox(Vector(0, 0, 0), Vector(3000, 3500, 20)))
    inputs_dataset.__enter__()
#    
#    input2 = TiffVolume(data_pth, BoundingBox(Vector(3001, 3501, 0), Vector(6000, 7000, 20)))
#    input2.__enter__()
#    
#    pooled_vol = PooledVolume(stack_size = 20)
#    pooled_vol.add(input1)    
#    pooled_vol.add(input2)        
#    
##    for f in os.listdir(data_pth):
##        tif = TiffVolume(os.path.join(data_pth, f), BoundingBox(Vector(0, 0, 0),
##                                                         Vector(1000, 1000, 1)))  # Create a volume to feed into predictor
##        tif.__enter__()
##        pooled_vol.add(tif)
#        
#    inputs = pooled_vol
    
    sys.stdout.write('*******************************************************************************\n\n\
           Starting predictions...\n\n') 
    
    outputs = Array(inputs_dataset.getBoundingBox().getNumpyDim()) #initialise output array
    
    # Setup a predictor for computing outputs
    predictor = Predictor(net, checkpoint='/jukebox/wang/zahra/conv_net/training/20181009_zd_train/models/model715000.chkpt', gpu_device=0) #zmd added gpu device

    predictor.run(pooled_vol, outputs, batch_size=6)  # Run prediction

    sys.stdout.write('*******************************************************************************\n\n\
           Finishing predictions :) Saving... \n\n') 
    
    tifffile.imsave("test_prediction.tif", pooled_vol.get().getArray().astype(np.float32)) #saves image output, zeros = cells??? so bizarre
    
    sys.stdout.write('*******************************************************************************\n\n\
            Saved!')
    
if __name__ == '__main__':
    
    main()
    


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