#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:17:43 2018

@author: wanglab
"""

from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    
    raw = os.listdir('/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages/'); raw.sort()
    label = os.listdir('/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputLabelImages/'); label.sort()
    
    
    raw_train, raw_test, label_train, label_test = train_test_split(raw, label, test_size = 0.3, train_size = 0.7, random_state = 1)
    
    raw_val, raw_test, label_val, label_test = train_test_split(raw_test, label_test, test_size = 0.666, train_size = 0.333, random_state = 1)