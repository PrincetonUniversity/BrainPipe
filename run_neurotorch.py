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
import numpy as np
from skimage.external import tifffile

def main():
    
    print ('\n\n      Using torch version: {}\n\n'.format(torch.__version__)) #zmd added 
    
    net = RSUNet()  # Initialize the U-Net architecture

    inputs = TiffVolume('/home/wanglab/Documents/training_data/train/raw/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_626-675_03_inputRawImages.tif')  # Create a volume containing inputs
    labels = TiffVolume('/home/wanglab/Documents/training_data/train/label/20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_626-675_03_inputLabelImages-segmentation')  # Create a volume containing labels

    print ('**************************************************************************************\n\n      Inputs and Labels read correctly. Starting training...\n\n') #zmd added
    
    trainer = Trainer(net, inputs, labels,  # Setup a network trainer
                      max_epochs=100, gpu_device=0)

    # Set the trainer to add a checkpoint every 50 epochs
    trainer = CheckpointWriter(trainer, checkpoint_dir='.', 
                               checkpoint_period=5)

    trainer.run_training()  # Start training

    outputs = TiffVolume('/home/wanglab/Documents/python/3dunet_data/neurotorch_training_data/val/raw/JGANNOTATION_20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_626-675_00_inputRawImages.tif') # Create a volume for predictions

    print ('*********************************************************************************************\n\n\
           Finished training :) Starting predictions...\n\n') #zmd added
    
    # Setup a predictor for computing outputs
    predictor = Predictor(net, checkpoint='./iteration_100.ckpt', gpu_device=0) #zmd added gpu device

    predictor.run(inputs, outputs, batch_size=1)  # Run prediction

    print ('*********************************************************************************************\n\n\
           Finishing predictions :) Saving... \n\n') #zmd added
    
    # Save outputs - zmd modified
    print (outputs.getArray())  #print probability array
    tifffile.imsave('outputs.tif', outputs.getArray()) #save output array as tiff

if __name__ == '__main__':
    main()