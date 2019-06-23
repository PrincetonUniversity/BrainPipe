#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:50:10 2017

@author: tpisano
"""
from skimage.util import view_as_windows
from tools.utils.overlay import tile
import scipy.stats
from scipy.ndimage.interpolation import zoom
from skimage.external import tifffile
import collections, numpy as np

if __name__ == '__main__':

    val = 500
    src = np.arange(val**2).reshape(val, val)
    #src = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero/20161204_tp_bl6_lob45_ml_03_nospread/full_sizedatafld/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_ch02/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_C02_Z1010.tif')
    dct = make_patches(src, patchsize=(250,250), stepsize=(200,200))
    cnn_patches = dct['patches']
    out = reconstruct_image(blending=True, **dct)
    np.all(out == src)
    
    #3d vol with 2d patching
    val = 500
    src = np.arange(val**3).reshape(val, val, val)
    #src = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero/20161204_tp_bl6_lob45_ml_03_nospread/full_sizedatafld/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_ch02/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_C02_Z1010.tif')
    #src = np.asarray([src]*50)
    dct = make_patches(src, patchsize=(1, 250,250), stepsize=(1,200,200))
    cnn_patches = dct['patches']
    out = reconstruct_image(blending=True, **dct)
    np.all(out == src)    
   
    #3d with 3d patching <---NOT FUNCTIONAL
    val = 100
    src = np.arange(val**3).reshape(val, val, val)
    src = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero/20161204_tp_bl6_lob45_ml_03_nospread/full_sizedatafld/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_ch02/20161204_tp_bl6_lob45_ml_03_nospread_488_555_647_0005na_400msec_z3um_1hfds_C02_Z1010.tif')
    src = np.asarray([src]*50)
    dct = make_patches(src, patchsize=(10, 25,25), stepsize=(10,20,20))
    cnn_patches = dct['patches']
    out = reconstruct_image(blending=True, **dct)
    np.all(out == src)    

#%%
def make_patches(src, patchsize=(250,250), stepsize=200, dtype=False):
    '''Function to generate patches for CNN,
    
    USEFUL for training, might be better to use memory efficient datagen for use of neural net due to memory intensivity
    
    Assumes stepsize will be less than window size.
    patchsize and stepsize must be of same length
    
    Inputs:
        src: 2d np array or 3d numpy array (if desired to tile 3d array in 2d patchsize=(1,y,x))
        patchsize: tuple(Y by X)
        stepsize: int or tuple(Y by X)
        dtype (optional): change to particular dtype
        
    Returns:
        dct
            {'dtype':dtype,
            'patches':patches,
            'patches_dims_before_reshape':patches_dims_before_reshape,
            'image_dims':image_dims,
            'stepsize':stepsize,
            'padding':[ypad,xpad]}
            
        for patches only: 
            cnn_patches = dct['patches']
    '''
    patchsize = tuple(patchsize)
    stepsize = tuple(stepsize)
    if not isinstance(patchsize, tuple): patchsize=(patchsize,)
    if not isinstance(stepsize, tuple): stepsize=(stepsize,)
    if len(patchsize)==4 and len(stepsize)==3: patchsize = patchsize[1:]
    assert len(patchsize) == len(stepsize), 'patchsize and stepsize must be of same length'
    image_dims = src.shape
    if not dtype: dtype=src.dtype
    #determine if added padding is needed
    #pad by a patchsize on each side since view_as_window stops before getting to the end (this ensures everything is captured)
    if len(src.shape)==2:
        if not isinstance(patchsize, collections.Iterable): patchsize = [patchsize, patchsize]    
        src = np.pad(src, ((0,patchsize[0]), (0, patchsize[1])), 'constant')
        patches = view_as_windows(src, patchsize, stepsize)
        patches_dims_before_reshape = patches.shape
        patches = patches.reshape(patches_dims_before_reshape[0]*patches_dims_before_reshape[1], patches_dims_before_reshape[2], patches_dims_before_reshape[3])        
    
    #3d input and tiling in 2d due to patchsize in z=1
    elif len(src.shape)==3 and patchsize[0] == 1:
        if not isinstance(patchsize, collections.Iterable): patchsize = [patchsize, patchsize, patchsize]
        zpad = 0 if patchsize[0] == 1 else patchsize[0]
        src = np.pad(src, ((0,zpad),(0,patchsize[1]), (0, patchsize[2])), 'constant')
        patches = np.squeeze(view_as_windows(src, patchsize, stepsize))
        patches_dims_before_reshape = patches.shape
        patches = patches.reshape(patches_dims_before_reshape[0]*patches_dims_before_reshape[1]*patches_dims_before_reshape[2], patches_dims_before_reshape[3], patches_dims_before_reshape[4])
    #3d input and tiling in 3d due to patchsize in z>1 
    elif len(src.shape)==3 and patchsize[0] > 1:
        if not isinstance(patchsize, collections.Iterable): patchsize = [patchsize, patchsize, patchsize]
        zpad = 0 if patchsize[0] == 1 else patchsize[0]
        src = np.pad(src, ((0,zpad),(0,patchsize[1]), (0, patchsize[2])), 'constant')
        patches = np.squeeze(view_as_windows(src, patchsize, stepsize))
        patches_dims_before_reshape = patches.shape
        patches = patches.reshape(patches_dims_before_reshape[0]*patches_dims_before_reshape[1]*patches_dims_before_reshape[2], patches_dims_before_reshape[3], patches_dims_before_reshape[4], patches_dims_before_reshape[5])
    
    

    dct = {'dtype':dtype, 'patches':patches.astype(dtype), 'patchsize': patchsize, 'patches_dims_before_reshape':patches_dims_before_reshape, 'image_dims':image_dims, 'stepsize':stepsize, 'padding':patchsize}
    
    return dct

def reconstruct_image(blending=True, patches=False, dims=False, stepsize=False, ypad=False, xpad=False, **kwargs):
    '''Wrapping for description and defaults
    
    skimage's function to generate windows of patchsize with stepsize
    
    Input = dictionary from makepatches
    
    blending: False, none
              True or 'max': take maximum pixel value between to images
    
    '''
    
    if 'patches' in kwargs: patches=kwargs['patches']
    if 'image_dims' in kwargs: image_dims=kwargs['image_dims']
    if 'stepsize' in kwargs: stepsize=kwargs['stepsize']
    if 'padding' in kwargs: padding =kwargs['padding']
    if 'patchsize' in kwargs: patchsize =kwargs['patchsize']
    if 'patches_dims_before_reshape' in kwargs: patches_dims_before_reshape =kwargs['patches_dims_before_reshape']
    
    if len(patchsize)==4 and len(stepsize)==3: patchsize = patchsize[1:]
    
    patches = patches.reshape(patches_dims_before_reshape)
    dtype = patches.dtype
    out = np.zeros(image_dims)
    
    if len(image_dims)==2:
        y,x = patchsize
        if not isinstance(stepsize, collections.Iterable): stepsize = [stepsize, stepsize]
        ystep, xstep = stepsize
        
        if not blending:
            for ytile in range(patches.shape[0]):
                for xtile in range(patches.shape[1]):
                    im = patches[ytile, xtile]
                    d0,d1 = out[ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x].shape
                    out[ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x] = im[:d0,:d1]
        
        else:
            for ytile in range(patches.shape[0]):
                for xtile in range(patches.shape[1]):
                    im = patches[ytile, xtile]
                    outim = out[ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x]
                    d0,d1 = outim.shape
                    out[ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x] = np.maximum(im[:d0,:d1], outim)
            
        #out = out[:-padding[0], :-padding[1]].astype(dtype)
        out = out.astype(dtype)
        
    elif len(image_dims)==3:
        z,y,x = patchsize
        if not isinstance(stepsize, collections.Iterable): stepsize = [stepsize, stepsize, stepsize]
        zstep, ystep, xstep = stepsize
        
        if not blending:
            for ztile in range(patches.shape[0]):
                for ytile in range(patches.shape[1]):
                    for xtile in range(patches.shape[2]):
                        im = patches[ztile, ytile, xtile]
                        if len(im.shape)==2: im = np.expand_dims(im, axis=0)
                        d0,d1,d2=out[ztile*zstep:(zstep*ztile)+z, ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x].shape
                        out[ztile*zstep:(zstep*ztile)+z, ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x] = im[:d0,:d1,:d2]
        
        else:
            for ztile in range(patches.shape[0]):
                for ytile in range(patches.shape[1]):
                    for xtile in range(patches.shape[2]):
                        im = patches[ztile, ytile, xtile]
                        if len(im.shape)==2: im = np.expand_dims(im, axis=0)
                        outim = out[ztile*zstep:(zstep*ztile)+z, ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x]
                        d0,d1,d2=outim.shape
                        out[ztile*zstep:(zstep*ztile)+z, ytile*ystep:(ystep*ytile)+y, xtile*xstep:(xstep*xtile)+x] = np.maximum(im[:d0,:d1,:d2], outim)
            
        #out = out[:-padding[0], :-padding[1], :-padding[2]].astype(dtype)
        out = out.astype(dtype)
    
    return out


#%%old likely can delete:
def patch_data(src, verbose = True, clean=True, **kwargs):
    '''
    
    Inputs
    ---------------
    src: memmapped array generated by generate_mem_mapped_array
    fov (either in kwargs or provided)
    
    Returns
    ----------
    patch_dct
    '''   
    #predict using cnn - might be different for cnn dimensions
    #80 gigs need for 25 gig dataset
    if verbose: sys.stdout.write('Loading dataset and generating a memory mapped array...'); sys.stdout.flush()
    arr = load_memmap_arr(src) 
    
    #patch up data
    if verbose: sys.stdout.write('...done.\nPatching data (this is memory intensive)...'); sys.stdout.flush(); start=time.time()
    patch_dct = make_patches(arr, kwargs['patchsize'], kwargs['stepsize'], dtype='float32')
    
    #save to mem_map
    if verbose: sys.stdout.write('...done in {} minutes.\nSaving out patches...'.format(np.round((time.time() - start)/60,2))); sys.stdout.flush()
    patches = src[:-4]+'_patches.npy'
    patcharr = load_memmap_arr(patches, mode = 'w+', shape = patch_dct['patches'].shape, dtype = 'float32')
    patcharr[:] = patch_dct['patches']
    patcharr.flush(); del patcharr
    
    #replace data with memmap and return dct
    patch_dct['patches'] = patches    
    return patch_dct
