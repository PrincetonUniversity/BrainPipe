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
    
    print ('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #zmd added 
    
    net = torch.nn.DataParallel(RSUNet())  # Initialize the U-Net architecture - zmd modified

    inputs = TiffVolume('/home/wanglab/Documents/training_data/train/raw/20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1350-1700_x3100-3450_inputRawImages.tif')  # Create a volume containing inputs
    inputs.__enter__()
    
#    labels = TiffVolume('/home/wanglab/Documents/training_data/train/label/20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y1350-1700_x3100-3450_inputLabelImages-segmentation.tif')  # Create a volume containing labels
#
#    print ('**************************************************************************************\n\n      Inputs and Labels read correctly. Starting training...\n\n') #zmd added
#    
#    trainer = Trainer(net, inputs, labels,  # Setup a network trainer
#                      max_epochs=100, gpu_device=0)
#
#    # Set the trainer to add a checkpoint every 50 epochs
#    trainer = CheckpointWriter(trainer, checkpoint_dir='.', 
#                               checkpoint_period=5)
#
#    trainer.run_training()  # Start training

    outputs = TiffVolume('/home/wanglab/Documents/python/NeuroTorch/data') # Create a volume for predictions
    outputs.__enter__()
    
    print ('*********************************************************************************************\n\n\
           Finished training :) Starting predictions...\n\n') #zmd added
    
    # Setup a predictor for computing outputs
    predictor = Predictor(net, checkpoint='/jukebox/wang/zahra/conv_net/20181009_zd_train/models/model715000.chkpt', gpu_device=0) #zmd added gpu device

    predictor.run(inputs, outputs, batch_size=1)  # Run prediction

    print ('*********************************************************************************************\n\n\
           Finishing predictions :) Saving... \n\n') #zmd added
    
#    print (outputs.get().getArray())
#    tifffile.imsave('outputs.tif', outputs.get().getArray())    
    
if __name__ == '__main__':
    main()