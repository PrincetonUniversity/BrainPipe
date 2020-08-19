#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:05:56 2016

@author: tpisano


NOTE WITH CROPPING THIS DOES NOT YET HAVE AN EFFECT ON THE ANNOTATION FILE....THIS NEEDS TO BE INCOROPORATED
"""
import os, numpy as np
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs, change_bitdepth
from tools.utils.directorydeterminer import directorydeterminer
import tifffile

def mask_atlas(dct = None, atlasfile=None, verbose=False, **kwargs):
    '''Helper function to mask atlas.
    
    Assumes zero based numerics: i.e. that the first entry in a dimension is 0 NOT 1.
    Uses numpy's ZYX convention
    
    Inputs:
    -----------------------
    dct (optional): dictionary consisting of x,y,z ranges to keep.
        NOTE: ranges to keep must be 
        e.g.
        dct = {'x': all, 'y': '125:202', 'z': '75:125'}
    atlasfile (optional): path to atlas file
    
    Returns:
    -----------------------
    atlas: np array of masked atlas
    '''
    
    #handle inputs    
    if not dct:
        dct = kwargs['maskatlas']
    
    #load atlas
    if not atlasfile:
        atlas = tifffile.imread(kwargs['AtlasFile'])
    else:
        atlas = tifffile.imread(atlasfile)
    
    if verbose: print ('Generating a mask of atlas\n   Keeping dims: {}'.format(dct))
    
    #works through each dimension
    for ii in range(3):
        
        #set dimension
        dim = ['x', 'y', 'z'][ii]
        
        #if dimension is not present keep all
        if not dim in dct: dct[dim] = ':'
        if str(dct[dim]) == '<built-in function all>': dct[dim] = ':'
        
    #generate mask of areas to zero out
    mask = np.ones(atlas.shape, dtype=bool)
        
    #keep areas within indices            
    exec('mask[{},{},{}]=False'.format(dct['z'], dct['y'], dct['x']))
    atlas[mask] = 0


    return atlas
    
def generate_masked_atlas(binarize = False, **kwargs):
    '''Wrapper to call mask_atlas and save in appropriate place
    
    Inputs
    -------------------
    binarize: (optional) if true convert all nonzero pixels to 255
    kwargs used: 
        ouputdirectory
        AtlasFile
        maskatlas
        
    Returns:
    -------------------
    path to newly saved masked atlas
    '''
    #make folder to store masked atlas
    fld = os.path.join(kwargs['outputdirectory'], 'maskedatlas')
    makedir(fld)
    
    #generate masked_atlas
    masked_atlas = mask_atlas(**kwargs)
    if binarize: 
        masked_atlas[masked_atlas>0]=255
        masked_atlas = change_bitdepth(masked_atlas)
    
    #save masked_atlas
    masked_atlas_pth = os.path.join(fld, 'masked_atlas.tif')
    tifffile.imsave(masked_atlas_pth, masked_atlas)
    
    return masked_atlas_pth
    
#%%
def crop_atlas(dct = None, atlasfile=None, verbose=False, **kwargs):
    '''Helper function to crop atlas.
    
    Assumes zero based numerics: i.e. that the first entry in a dimension is 0 NOT 1.
    Uses numpy's ZYX convention
    
    Inputs:
    -----------------------
    dct (optional): dictionary consisting of x,y,z ranges to keep.
        NOTE: ranges to keep must be 
        e.g.
        dct = {'x': all, 'y': '125:202', 'z': '75:125'}
    atlasfile (optional): path to atlas file
    
    Returns:
    -----------------------
    atlas: np array of cropped atlas
    '''
    
    #handle inputs    
    if not dct:
        dct = kwargs['cropatlas']
    
    #load atlas
    if not atlasfile:
        atlas = tifffile.imread(kwargs['AtlasFile'])
    else:
        atlas = tifffile.imread(atlasfile)
    
    if verbose: print ('Generating a cropped atlas\n   Keeping dims: {}'.format(dct))
    
    #works through each dimension
    for ii in range(3):
        
        #set dimension
        dim = ['x', 'y', 'z'][ii]
        
        #if dimension is not present keep all
        if not dim in dct: dct[dim] = ':'
        if str(dct[dim]) == '<built-in function all>': dct[dim] = ':'
        
    #crop areas
    crop = np.ones(atlas.shape, dtype=bool)
    
    #keep areas within indices            
    exec('crop = atlas[{},{},{}]'.format(dct['z'], dct['y'], dct['x']))    
    #atlas[mask] = 0


    return crop
    
def generate_cropped_atlas(**kwargs):
    '''Wrapper to call crop_atlas and save in appropriate place
    
    Inputs
    -------------------
    kwargs used: 
        ouputdirectory
        AtlasFile
        maskatlas
        
    Returns:
    -------------------
    path to newly saved cropped atlas
    '''
    #make folder to store masked atlas
    fld = os.path.join(kwargs['outputdirectory'], 'croppedatlas')
    makedir(fld)
    
    #generate masked_atlas
    cropped_atlas = crop_atlas(**kwargs)
    
    #save masked_atlas
    cropped_atlas_pth = os.path.join(fld, 'cropped_atlas.tif')
    tifffile.imsave(cropped_atlas_pth, cropped_atlas)
    
    return cropped_atlas_pth
    

    
    
    
    
    
    
    
    