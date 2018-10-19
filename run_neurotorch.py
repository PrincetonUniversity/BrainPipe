#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:14:05 2018

@author: wanglab
"""

from neurotorch.datasets.filetypes import TiffVolume
from neurotorch.core.trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.core.predictor import Predictor
from neurotorch.training.checkpoint import CheckpointWriter
import torch
from skimage.external import tifffile

def main():
    
    print ('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct
    
    net = torch.nn.DataParallel(RSUNet())  # Initialize the U-Net architecture

    inputs = TiffVolume('/home/wanglab/Documents/python/NeuroTorch/data')  # Create a volume to feed into predictor
    inputs.__enter__()

#    outputs = TiffVolume('/home/wanglab/Documents/python/NeuroTorch/data') # Create a volume for predictions
#    outputs.__enter__()
    
    print ('*******************************************************************************\n\n\
           Finished training :) Starting predictions...\n\n') 
    
    # Setup a predictor for computing outputs
    predictor = Predictor(net, checkpoint='/jukebox/wang/zahra/conv_net/20181009_zd_train/models/model715000.chkpt', gpu_device=0) #zmd added gpu device

    predictor.run(inputs, outputs, batch_size=1)  # Run prediction

    print ('*******************************************************************************\n\n\
           Finishing predictions :) Saving... \n\n') #zmd added
    
#    print (outputs.get().getArray())
#    tifffile.imsave('outputs.tif', outputs.get().getArray())    
    
if __name__ == '__main__':
    main()