#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:54:23 2018

@author: tpisano
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:47:10 2018

@author: tpisano
"""
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
#   3D plot of the
import matplotlib.pyplot as plt
from skimage.morphology import ball, cube, star, octagon
from tools.conv_net.functions.dilation import dilate_with_element, cylinder
import math, cv2
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import ndimage as ndi

if __name__ == '__main__':
    
    #goal is to generate noise, ask if objects are linear or spherical and exclude based on some feature
    dims = (1,100,100,100)
    src = dilate_with_element(generate_random_points(dims, numpoints=5)[0], ball(3))
    for r in range(5,13,2):    
        src = np.maximum(src, dilate_with_element(generate_random_points(dims, numpoints=5)[0], ball(r)))
    lbl = ndi.label(src)
    centers = ndi.measurements.center_of_mass(src, lbl[0], range(1, lbl[1]+1))
    plt.imshow(np.max(src,0))
    plt.imshow(np.max(lbl[0],0))
    val, cnts = np.unique(lbl[0], return_counts=True)
    labels = size_filter(lbl, lower=200, upper=50000)
    #%timeit size_filter(lbl, lower=200, upper=50000)
    plt.imshow(np.max(labels[0],0))

#%%
def size_filter(labels, lower=50, upper=50000):
    '''Function filter based on size
    '''
    
    vals, cnts = np.unique(labels[0], return_counts=True)
    for i, val in enumerate(vals):
        if cnts[i]<lower: labels[0][labels[0]==val]=0
        if cnts[i]>upper: labels[0][labels[0]==val]=0
    return labels

def generate_random_points(dims, numpoints):
    arr = np.zeros(dims).astype('bool')
    pnts = np.asarray([[np.random.randint(dims[yy]) for yy in range(len(dims))] for xx in range(numpoints)])
    for pnt in pnts:
        arr[pnt[0], pnt[1], pnt[2], pnt[3]]=True
    return arr