#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:56:45 2017

@author: tpisano
"""
import numpy as np
from math import sqrt, exp
from skimage.morphology import disk, dilation, binary_dilation, ball
from tools.imageprocessing.target_arrays import generate_target_array_selem, apply_dilation



#to increase efficiency make this so you generate the element once and then apply to all

if __name__ == '__main__':
    
    #init sample data
    arr = np.zeros((250,250,250)).astype('bool')
    pnts = np.asarray([(np.random.randint(250), np.random.randint(250), np.random.randint(250)) for xx in range(10)])
    for pnt in pnts:    
        arr[pnt[0], pnt[1], pnt[2]]=True
    
    #cylinder
    selem = cylinder(radius=9, depth=13)
    #%timeit dilate_with_element(arr, selem)
    out = dilate_with_element(arr, selem)
    plt.imshow(np.max(out, 0))
    
    #sphere with changing intensity
    selem = sphere_changing_intensity(radius=14, decayfactor=9)
    #%timeit dilate_with_element(arr, selem)
    out = dilate_with_element(arr, selem)
    plt.imshow(np.max(out*1000, 0))
    
    #target_array
    #%timeit apply_dilation(arr)
    #plt.imshow(np.max(out, 0))
    #more efficient target_array
    selem = generate_target_array_selem()
    #%timeit dilate_with_element(arr, selem)
    out = dilate_with_element(arr, selem)
    plt.imshow(np.max(out, 0))
        
def dilation_wrapper(src, **kwargs):
    '''
    '''
    selem = kwargs['dilationfunction'](*kwargs['paramlist'])
    return dilate_with_element(src, selem)
    
    
def generate_arr_from_pnts_and_dims(pnts, dims=(40, 7422, 6262)):
    '''Function to generate array of zeros excpet where points
    '''
    arr = np.zeros(dims).astype('bool')
    for pnt in pnts:    
        arr[pnt[0], pnt[1], pnt[2]]=True
    return arr
    
    
def dilate_with_element(src, selem):
    '''Function to dilate pixels given a structuring element
    '''
    assert np.all([True for xx in selem.shape if (xx-1)%2]), 'structuring element must have odd dimensions'
    
    #find nonzero points before any changes to src - done for mem efficiency
    pnts = np.asarray(np.nonzero(src)).T
    
    #convert to bool -- will be faster
    src = np.zeros_like(src).astype('float32')
    
    #find deltas
    delta = [(xx-1)/2 for xx in selem.shape]
    
    for pnt in pnts:    
        #find deltas
        lowerdelta = [xx-yy for xx,yy in zip(pnt, delta)]
        upperdelta = [xx+yy+1 for xx,yy in zip(pnt, delta)]
        
        #check to see if edge cases 
        zls, yls, xls = [abs(min(xx, 0)) for xx in lowerdelta]
        zus, yus, xus = [abs(max(xx-yy,0)) for xx, yy in zip(upperdelta, src.shape)]
        
        #get abs for numpy
        zl,yl,xl = [abs(max(xx, 0)) for xx in lowerdelta]
        zu,yu,xu = [abs(min(xx, yy)) for xx, yy in zip(upperdelta, src.shape)]
        
        #apply
        zs, ys, xs = selem.shape
        im = src[zl:zu, yl:yu, xl:xu]
        src[zl:zu, yl:yu, xl:xu] = np.maximum(selem[zls:zs-zus, yls:ys-yus, xls:xs-xus], im)

    return norm(src)
    
def cylinder(radius, depth):
    '''Function to generate a cylinder
    '''
    return np.stack([disk(radius, dtype='bool') for i in range(depth)], axis=0)

def sphere_changing_intensity(radius=10, decayfactor=2):
    '''
    '''
    assert radius%2 == 0, 'radius must be an even number'
    radii = range(radius)
    vals = np.logspace(0.1, 1, len(radii))
    out = np.zeros([2*radius+5]*3).astype('float32')
    out[tuple([radius+3]*3)]=1
    src = np.copy(out)
    for i, r in enumerate(radii):
        out = out+dilate_with_element(src, ball(r)) * vals[i]
    out = out**decayfactor
    return norm(out)
    
def norm(src):
    return (src - np.min(src)) / (np.max(src) - np.min(src))