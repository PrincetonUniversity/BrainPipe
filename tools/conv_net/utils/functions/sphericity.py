#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:47:10 2018

@author: tpisano
"""
import numpy as np, cv2
import matplotlib.pyplot as plt
from skimage.morphology import ball, cube, star, octagon
from tools.conv_net.functions.dilation import dilate_with_element, cylinder
from scipy import ndimage as ndi

def _array(src, value):
    '''Generates small array from larger array where values are equal
    '''
    z,y,x = np.where(src==value)
    out = np.copy(src[np.min(z):np.max(z), np.min(y):np.max(y), np.min(x):np.max(x)]) #copy is critical
    out[out!=value]=0
    return out.astype('uint8')


def filter_sphericity(labels, cutoff=0.5, dims=3):
    '''Function that takes in from scipy.ndimage.labels and a cutoff value between 0 and 1
    
    1 is more spherical
    dims=(2,3) number of dimensions to look at

    '''
    vals = range(0, labels[1]+1) #zero indexing
    if len(vals) == 1 and vals[0] == 0: #np array of zeros
        return labels
    else:
        for val in vals[1:]:
            try:
              if sphericity(_array(labels[0], val), dims) < cutoff: labels[0][labels[0]==val]=0
            except Exception as e:
              print(e, val, 'Sphericity', dims, cutoff)
              #print 'if need more than two values to unpack error here upgrade to cv2 3+'
              #np.save('/jukebox/wang/pisano/bad_tvol_{}.npy'.format(str(np.random.randint(1000)).zfill(4)), labels[0])
              #print _array(labels[0], val), '_111_'
              #print sphericity(_array(labels[0], val), dims=dims), '_222_'
              #pass
              raise Exception
        return labels


def filter_sphericity_slow(labels, cutoff=0.5):
    '''Function that takes in from scipy.ndimage.labels and a cutoff value between 0 and 1
    
    1 is more spherical
    '''
    #labels = tuple((np.copy(labels[0]), labels[1]))
    for val in range(1, labels[1]+1):
        tvol = np.zeros_like(labels[0])
        tvol[labels[0]==val]=1
        if sphericity(tvol) < cutoff: labels[0][labels[0]==val]=0
    return labels
        
# filter contours
def sphericity(src, dims=3):
    '''src = 3d
    looks at it from two perspectives and then takes the total average
    dims=(2,3) number of dimensions to look at
    
    ball(9) = .895
    cube(9) = .785
    cylinder(9,9) = .770
    np.asarray([star(9) for xx in range(9)]) = .638
    
    sometime two contours are found on a zplane after labels - in this case take min, but could take average?
    '''
    lst = []; #src = np.copy(src).astype('uint8')
    if 0 in src.shape: return 0 #projects against empty labels
    for z in src:
        contours = findContours(z)
        circ = circularity(contours)
        if len(circ)>0: lst.append(np.min(circ))
    if dims==3:
        for z in np.swapaxes(src, 0,1):
            contours = findContours(z)
            circ = circularity(contours)
            if len(circ)>0: lst.append(np.min(circ))
    out = np.asarray(lst)
    if len(out)>0: 
        return np.mean(out)
    else:
        return 0

def circularity(contours):
    """
    A Hu moment invariant as a shape circularity measure, Zunic et al, 2010
    """
    #moments = [cv2.moments(c.astype(float)) for c in contours]
    #circ = np.array([(m['m00']**2)/(2*np.pi*(m['mu20']+m['mu02'])) if m['mu20'] or m['mu02'] else 0 for m in moments])
    circ = [ (4*np.pi*cv2.contourArea(c))/(cv2.arcLength(c,True)**2) for c in contours]

    return np.asarray(circ)

def findContours(z):
    '''Function to handle compatiblity of opencv2 vs 3
    '''
    if str(cv2.__version__)[0] == '3':
        cim,contours,hierarchy = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #if need more than two values to unpack error here upgrade to cv2 3+
    elif str(cv2.__version__)[0] == '2':
        contours,hierarchy = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #if need more than two values to unpack error here upgrade to cv2 3+
    contours = np.asarray([c.squeeze() for c in contours if cv2.contourArea(c)>0])
    return contours

def generate_random_points(dims, numpoints):
    arr = np.zeros(dims).astype('bool')
    pnts = np.asarray([[np.random.randint(dims[yy]) for yy in range(len(dims))] for xx in range(numpoints)])
    for pnt in pnts:
        arr[pnt[0], pnt[1], pnt[2], pnt[3]]=True
    return arr

def get_sphericity(labels, centers, dims=3, box_size=75):
    '''Function that takes in from scipy.ndimage.labels
    
    1 is more spherical
    dims=(2,3) number of dimensions to look at
    box_size: 1/2 size of bounding box (should be comfortably larger than cell). 50 = 50pixels +/- center, meaning a 100pixel length cube

    '''
    if labels[1]==0: #np array of zeros
        return labels
    else:
        vals = range(1, labels[1]+1) #don't need zero and account for zero indexing
        sphericity_values=[]
        for val,center in zip(vals, centers):
            try:
              sphericity_values.append(sphericity(bounding_box_from_center_array(labels[0], val, center, box_size=box_size), dims))
            except Exception as e:
              print(e, val, 'Sphericity', dims)
              #print 'if need more than two values to unpack error here upgrade to cv2 3+'
              #np.save('/jukebox/wang/pisano/bad_tvol_{}.npy'.format(str(np.random.randint(1000)).zfill(4)), labels[0])
              #print _array(labels[0], val), '_111_'
              #print sphericity(_array(labels[0], val), dims=dims), '_222_'
              #pass
              raise Exception
        return np.asarray(sphericity_values)

def bounding_box_from_center_array(src, val, center, box_size=75):
    '''Faster version of _array as it makes a box rather than calculating
    '''
    z,y,x = [int(xx) for xx in center]
    out = np.copy(src[max(0,z-box_size):z+box_size, max(0,y-box_size):y+box_size, max(0,x-box_size):x+box_size]) #copy is critical
    out[out!=val]=0
    return out.astype('uint8')



if __name__ == '__main__':
    
    #goal is to generate noise, ask if objects are linear or spherical and exclude based on some feature
    dims = (1,500,500,500)
    
    #choice of obejct
    selem = ball(9)
    selem = cube(9)
    selem = cylinder(9,9)
    selem = np.asarray([star(9) for xx in range(9)])
    
    #dilate
    src = dilate_with_element(generate_random_points(dims, numpoints=20)[0], selem)
    lbl = ndi.label(src)
    centers = ndi.measurements.center_of_mass(src, lbl[0], range(1, lbl[1]+1))
    plt.imshow(np.max(src,0))
    plt.imshow(np.max(lbl[0],0))
    
    ##
    lbl_vol = np.copy(lbl[0])
    lbl_vol[lbl[0]>0] = 1
    sitk.Show(sitk.GetImageFromArray(lbl_vol))
    
    for val in range(1, lbl[1]+1):
        tvol = np.zeros_like(lbl[0])
        tvol[lbl[0]==val]=1
        plt.imshow(np.max(tvol, 0))
        sphericity(tvol)
        #%timeit sphereness(tvol)
        
        
    #%timeit filter_sphericity_slow(lbl, cutoff=0.5)
    #%timeit filter_sphericity(lbl, cutoff=0.5)
    out = filter_sphericity(lbl, cutoff=0.75, dims=3)
    plt.imshow(np.max(out[0],0))
    
    #to get sphericity values
    #time sphericity_values = get_sphericity(lbl, centers, dims=3,  box_size=75)
    box_size = 75 #1/2 size of bounding box (should be comfortably larger than cell). 50 = 50pixels +/- center, meaning a 100pixel length cube

    sphericity_values = get_sphericity(lbl, centers, dims=3)
    