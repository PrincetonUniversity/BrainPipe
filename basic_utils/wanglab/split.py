#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:00:47 2018

@author: wanglab
"""

from sklearn.model_selection import train_test_split
import os, shutil

data_pth = '/home/wanglab/Documents/training_data'
pth = '/home/wanglab/Documents/inputRawImages'
lbl = '/home/wanglab/Documents/inputLabelImages'

#split into train, val, and test data
raw = os.listdir(pth)
raw = [xx[:-19] for xx in raw] #because we add the extension back in sampler, can change later
#-18 for h5s, -19 for tifs
#FIXME - do regex

#%70-20-10 train-validation-test
train, test = train_test_split(raw, test_size = 0.3, train_size = 0.7, random_state = 1)   
val, test = train_test_split(test, test_size = 0.333, train_size = 0.666, random_state = 1)

#FIXME: messy, make it easier to do without changing all paths
#test
for i in test: #move raw and labels to appropriate bins
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    drc = data_pth+'/test/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
    
for i in test:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    drc = data_pth+'/test/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
    
#train
for i in train: 
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    drc = data_pth+'/train/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
    
for i in train:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    drc = data_pth+'/train/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
    
#val
for i in val: 
    t = os.path.join(lbl, i+'_inputLabelImages-segmentation.tif')
    drc = data_pth+'/val/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
    
for i in val:
    t = os.path.join(pth, i+'_inputRawImages.tif')
    drc = data_pth+'/val/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.move(t, drc)
