#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:00:47 2018

@author: wanglab
"""

from sklearn.model_selection import train_test_split
import os, shutil

#doing this so i can remember which datasets are in which category - makes it easier to have physical copies

data_pth = "/home/wanglab/LightSheetData/rat-brody/processed/201910_tracing/training/inputs"
if not os.path.exists(data_pth): os.mkdir(data_pth)

pth = "/home/wanglab/LightSheetData/rat-brody/processed/201910_tracing/training/otsu"

#split into train, val, and test data
raw = [xx[:-8] for xx in os.listdir(pth) if xx[-8:] == "_img.tif"] #cutoff extension coz will add it later with labels

#%70-20-10 train-validation-test
train, test = train_test_split(raw, test_size = 0.3, train_size = 0.7)   
val, test = train_test_split(test, test_size = 0.333, train_size = 0.666)

#FIXME: messy, make it easier to do without changing all paths but safer
#test
for i in test: #move raw and labels to appropriate bins
    t = os.path.join(pth, i+'_lbl.tif')
    drc = data_pth+'/test/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)
    
for i in test:
    t = os.path.join(pth, i+'_img.tif')
    drc = data_pth+'/test/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)

print("copied files into test directory")    
#train
for i in train: 
    t = os.path.join(pth, i+'_lbl.tif')
    drc = data_pth+'/train/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)
    
for i in train:
    t = os.path.join(pth, i+'_img.tif')
    drc = data_pth+'/train/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)

print("copied files into train directory")        
#val
for i in val: 
    t = os.path.join(pth, i+'_lbl.tif')
    drc = data_pth+'/val/label'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)
    
for i in val:
    t = os.path.join(pth, i+'_img.tif')
    drc = data_pth+'/val/raw'
    if not os.path.exists(drc): os.makedirs(drc)
    shutil.copy(t, drc)
print("copied files into val directory")    
