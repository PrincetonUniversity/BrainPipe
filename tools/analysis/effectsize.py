#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:20:39 2017

@author: tpisano
"""

import numpy as np, matplotlib.pyplot as plt

def cohen_d(x,y, log_transform=False):
    '''Function to calculate effect size based on https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
    
    x,y = are iterables representing dataset
    log_transform: true to take log of data. NOTE ZEROS IN THIS SAMPLE ARE DROPPED****
    
        
    '''
    from numpy import std, mean, sqrt
    if log_transform:
        if 0 in x: 
            print('Dropping zeros from dataset because logtransform is set to True, you might need to check data as to why you have zeros')
            x=[xx for xx in x if xx !=0]
        if 0 in y: 
            print('Dropping zeros from dataset because logtransform is set to True, you might need to check data as to why you have zeros')
            x=[xx for xx in y if xx !=0]
        x = np.log10(x)
        y = np.log10(y)
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

if __name__ == '__main__':
    #a = np.random.random_sample(20)
    #b = np.random.random_sample(440)
    a = np.random.normal(20, size=440)
    b = np.random.normal(20, size=450)
    c = a + 0.2
    
    #compensating for sample size
    aa = cohen_d(a, a)
    print(aa)
    abp = cohen_d(a,b)
    ab = cohen_d(a,b, pool=False)
    print(abp, ab)
    acp = cohen_d(c, a, plot=True)
    ac = cohen_d(c, a, pool=False)
    print(acp, ac)
    #plt.hist(a); plt.hist(b); plt.hist(c)
    m0 = [9, 7, 8, 9, 8, 9, 9, 10, 9, 9]
    m1 = [9, 6, 7, 8, 7, 9, 8, 8, 8, 7]
    #s0 = np.std(m0, ddof=1)
    #s1 = np.std(m1, ddof=1)
    cohen_d(m0, m1)
    cohen_d(m0, m1, log_transform=True)
