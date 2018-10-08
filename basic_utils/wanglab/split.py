#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:00:47 2018

@author: wanglab
"""

from sklearn.model_selection import train_test_split
import os, shutil

pth = '/home/wanglab/Documents/python/data/training_data/inputRawImages'
lbl = '/home/wanglab/Documents/python/data/training_data/inputLabelImages'

#split into train, val, and test data
raw = os.listdir(pth)
raw = [xx[:-19] for xx in raw] #because we add the extension back in sampler, can change later
#-18 for h5s, -19 for tifs
#FIXME - do regex

train, test = train_test_split(raw, test_size = 0.3, train_size = 0.7, random_state = 1)   
val, test = train_test_split(test, test_size = 0.666, train_size = 0.333, random_state = 1)

#FIXME: messy, make it easier to do without changing all paths
#test
for i in test: #move raw and labels to appropriate bins
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/test/label')
    
for i in test:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/test/raw')
    
#train
for i in train: 
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/train/label')
    
for i in train:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/train/raw')
    
#val
for i in val: 
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/val/label')
    
for i in val:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    shutil.move(t, '/home/wanglab/Documents/python/data/training_data/val/raw')