#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:11:22 2016

@author: tpisano
"""

import numpy as np, SimpleITK as sitk, sys
import tifffile
import itertools
import matplotlib.pyplot as plt

####################################################################################################
########################Functions###################################################################
####################################################################################################
def fix_orientation(imstack, axes=None, verbose=False, **kwargs):
    '''Function to fix orientation of imaging series relative to atlas. Note assumes inputs of tuple of STRINGS**** not ints. 
    This allows for '-0' which is need if one wants to reverse the x axis.
    Assumes XYZ orientation ('0','1','2'). To reverse the order of an axis add a '-'. I.e. to flip the Y axis: ('0', '-1', '2').
    
    Order of operations is reversing of axes BEFORE swapping axes. E.g: ('2','-0','1'). This means reverse the X axis, then move: X->Y, Z->X, Y->Z.
    
    Inputs
    --------------------
        imstack: np.array of image
        axes (optional): tuple of strs ('0','1','2')
        verbose (optional): gives information about shape before and after
        kwargs (optional): will look for kwargs['finalorientation'] = ('0','1','2')
        
    Returns
    --------------------
        imstack: numpy array reorientated
        
    '''
    #handle inputs
    if not axes:
        try:
            axes = kwargs['finalorientation']
        except KeyError:
            #reload param dictionary from original run_tracing file and update kwargs
            import tools.utils.io as io
            sys.path.append(kwargs['packagedirectory'])
            import run_tracing
            kwargs.update(run_tracing.params)
            io.save_kwargs(**kwargs)
            #reload kwargs now that it has been updated
            kwargs = io.load_kwargs(kwargs['outputdirectory'])
            axes = kwargs['finalorientation']
    
    #verbosity:
    if verbose:
        shape_before = imstack.shape
        origax = axes
    
    #handle reversing of an axis
    imstack = reverse_axis(imstack, axes)
    
    #change from XYZ to ZYX (np's) convention and remove negatives from axes and change to ints now that axes have been flipped
    axes = [abs(2-abs(int(xx))) for xx in axes][::-1]
    
    #swap axes:
    imstack = imstack.transpose(axes)
    
    #verbosity:
    if verbose:
        sys.stdout.write('\n\nfix_orientation function:\n   "finalorientation" command (xyz): {}\n   Shape before(xyz): {}\n   Shape after(xyz): {}\n\n'.format(origax, shape_before[::-1], imstack.shape[::-1])); sys.stdout.flush()
    return imstack
    
def reverse_axis(imstack, axes, verbose=False):
    '''helper function to reverse axes. Assumes xyz convention rather than numpys zyx convention
    
    Inputs are strings rather than integers allows for handling of '-0'
    '''
    #ensure inputs:
    assert type(axes) == tuple, 'axes must be a tuple of strings'
    assert not any([True for xx in axes if type(xx) != str]), 'axes must be a tuple of strings' #ensure all tuples entries are strings
    
    #check and reverse axis if negative. Remember that np's convention is z,y,x while input to this function assumes x,y,z
    for axis_to_reverse in axes:
        if axis_to_reverse == '-0':
            imstack = imstack[:,:,::-1]
        elif axis_to_reverse == '-1':
            imstack = imstack[:,::-1,...]
        elif axis_to_reverse == '-2':
            imstack = imstack[::-1,...]
    
    return imstack
    


def fix_contour_orientation(contour_array, axes=None, verbose = True, **kwargs):
    '''Function to fix orientation of contour array relative to atlas. Note assumes inputs of tuple of STRINGS**** not ints. 
    This allows for '-0' which is need if one wants to reverse the x axis.
    Assumes XYZ orientation ('0','1','2'). To reverse the order of an axis add a '-'. I.e. to flip the Y axis: ('0', '-1', '2').
    
    Order of operations is reversing of axes BEFORE swapping axes. E.g: ('2','-0','1'). This means reverse the X axis, then move: X->Y, Z->X, Y->Z.
    
    Inputs
    --------------------
        contour_array: np.array of contours. Stucture is array [number of cells, [z,y,x]] NOTE follows zyx convention from cell_class objects
        axes (optional): tuple of strs ('0','1','2')
        kwargs: (REQUIRES): will look for kwargs['finalorientation'] = ('0','1','2')
        
    Returns
    --------------------
        contour_array: numpy array reorientated

    '''  
    
    
    #handle inputs
    if not axes:
        axes = kwargs['finalorientation']
    
    #verbosity:
    if verbose:
        before = np.copy(contour_array[0:2])
        origax = axes
    
    #handle reversing of an axis
    contour_array = reverse_contour_axis(np.asarray(contour_array), axes, **kwargs)

    #change from XYZ to ZYX (np's) convention and remove negatives from axes and change to ints now that axes have been flipped
    axes = [abs(2-abs(int(xx))) for xx in axes][::-1]
    
    #swap axes:     #handle rotation in 3D:
    contour_array = swap_contour_axes(contour_array, axes)

        #verbosity:
    if verbose:
        sys.stdout.write('\n\nfix_orientation function:\n   "finalorientation" command (xyz): {}\n   Sample before(**zyx**): \n{}\n\n   Sample after(**zyx**): \n{}\n\n'.format(origax, before, contour_array[0:2])); sys.stdout.flush()
    
    ##FIXME: still need to make swap_contour_axes and then also deal with z,y,x dimensions after switching....
    #cellarr=swap_cols(cellarr, *kwargs['swapaxes']) ###change columns to account for orientation changes between brain and atlas: if horizontal to sagittal==>x,y,z relative to horizontal; zyx relative to sagittal
    #z,y,x=swap_cols(np.array([vol_to_process.fullsizedimensions]), *kwargs['swapaxes'])[0]##convert full size cooridnates into sagittal atlas coordinates
    #sys.stdout.write('Swapping Axes')

    return contour_array


def reverse_contour_axis(contour_array, axes, **kwargs):
    '''helper function to reverse axes. Assumes xyz convention rather than numpys zyx convention
    
    Inputs are strings rather than integers allows for handling of '-0'
    '''
    #ensure inputs:
    assert type(axes) == tuple, 'axes must be a tuple of strings'
    assert not any([True for xx in axes if type(xx) != str]), 'axes must be a tuple of strings' #ensure all tuples entries are strings
    
    #account for orientation differences between volume and atlas as cell detection happens in different processes: thus contours need to be fixed after
    #find full size dims:
    vol = kwargs['volumes'][0]
    zo, yo, xo = vol.fullsizedimensions

    #effectively reverse the order of each dimension if axes is negative. Axes convention is xyz, while array is zyx
    for i in range(len(contour_array)):
        for axis_to_reverse in axes:
            if axis_to_reverse == '-0':
                contour_array[i, 2] = xo - (contour_array[i, 2] + 1)
            elif axis_to_reverse == '-1':
                contour_array[i, 1] = yo - (contour_array[i, 1] + 1)
            elif axis_to_reverse == '-2':
                contour_array[i, 0] = zo - (contour_array[i, 0] + 1)

    return contour_array


def swap_contour_axes(contour_array, axes):
    '''Function to swap axes. Assumes that stack is array of [#ofcells, [z,y,x]]
    
    Note this follows zyx convention rather than xyz convention
    '''
    #ensure proper input
    assert(len(axes) ==3)
    assert not any([True for xx in axes if type(xx) != int]) #ensure all tuples entries are ints
    
    #make axes index pairs
    axes_index_pair = []
    [axes_index_pair.append((axes[index], index)) for index in range(3)]
    
    #remove axes in correct place already:
    todo = [xx for xx in axes_index_pair if xx[0] != xx[1]]; 
    
    #find unique moves:
    todo2 = list(set([tuple(sorted(xx)) for xx in todo]))
    
    #need to separate two cases of multiple axis flips, this because indexes of axes don't follow between multiple moves
    if axes == [1,2,0]:
        for swaps in [(0,1), (1,2)]:
            contour_array=swap_cols(contour_array, *swaps)
    if axes == [2,0,1]:
        for swaps in [(0,1), (0,2)]:
            contour_array=swap_cols(contour_array, *swaps)
    elif len(todo2) < 3:    
        for swaps in todo2:
            contour_array=swap_cols(contour_array, *swaps)
               

    return contour_array



def swap_cols(arr, frm, to):
    '''helper function used to swap np array columns if orientation changes have been made pre-registration
    '''
    arr[:, [frm, to]]=arr[:, [to, frm]]
    return arr


def fix_dimension_orientation(zyx=None, axes = None, verbose=False, **kwargs):
    '''Function to fix orientation of ***zyx*** (NOT XYZ) dimensions relative to atlas.

    Note:
    tuple input to function is ZYX while XYZ orientation in kwargs['finalorientation']...e.g.('0','1','2').
    

    Inputs
    --------------------
        (z,y,x) (optional): tuple of z,y,x dimensions    
        axes (optional): **XYZ*** tuple of strs ('0','1','2')
        kwargs: (optional): will look for kwargs['finalorientation'] = ('0','1','2')
        
    Returns
    --------------------
        z,y,x: tuple of ints of dimensions reorientated

    '''
    #ensure proper inputs
    if zyx: 
        assert type(zyx) == tuple
        z, y, x = zyx
    if not zyx:
        vol = kwargs['volumes'][0]
        z, y, x = vol.fullsizedimensions        
    if not axes:
        if 'finalorientation' in kwargs:
            axes = kwargs['finalorientation']
        else:
            raise Exception('Old software, update run_tracing_file' )

    #remove negatives from axes and change to ints since you don't need to handle reversing of axis for dimensions
    axes = [abs(int(xx)) for xx in axes]
    
    #verbosity
    if verbose:
        origax = np.copy(axes)
        old = [xx for xx in np.copy(zyx)]
        
    #reorient
    arr = swap_contour_axes(np.asarray([[z,y,x]]), axes)
    
    zyx = tuple([yy for xx in arr for yy in xx])
    
    #verbosity
    if verbose:
        sys.stdout.write('\n\nfix_dimension_orientation function:\n   "finalorientation" command (xyz): {}\n   Before(**zyx**): \n{}\n\n   After(**zyx**): \n{}\n\n'.format(origax, old, zyx)); sys.stdout.flush()
    
    return zyx

if __name__=='__main__':

    imstack = np.zeros((222,111,100)) #ZYX
    imstack[0:50,0:50,0:50]=255
    imstack[-50:,-50:,-50:]=100
    imstack[70:100,70:100, 60:80]=50
    plt.ion(); plt.figure(); plt.imshow(np.max(imstack, axis=0))
    plt.ion(); plt.figure(); plt.imshow(np.max(imstack, axis=1))
    plt.ion(); plt.figure();
    for axes in itertools.permutations((0,1,2)):
        print
        axes = tuple([str(xx) for xx in axes])
        #print axes
        newim = fix_orientation(imstack, axes=axes, verbose=True)
        plt.ion(); plt.figure(); plt.imshow(np.max(newim, axis=0))

        #print imstack.shape
        #print newim.shape
        
    imstack = tifffile.imread('/home/wanglab/LightSheetData/marm_ghazanfar/post_processed/marm_01/smallctx/marm_01_smallctx_488_647_90msec_z3um_20hfds_01na_resized_ch00_resampledforelastix.tif')
    kwargs={}
    #kwargs['finalorientation']= ('1', '-0','2')
    kwargs['finalorientation']= ('2','0','-1')
    newim = fix_orientation(imstack, axes=None, **kwargs)
    
    sitk.Show(sitk.GetImageFromArray(imstack))
    sitk.Show(sitk.GetImageFromArray(newim))
        
    from tools.utils.io import load_kwargs
    pth = '/home/wanglab/wang/pisano/tracing_output/bl6_ts/20150804_tp_bl6_ts01'
    kwargs = load_kwargs(pth)
   
    axes = ('-2','-0','-1')
    contour_array= np.zeros((2, 3))
    contour_array[0] = np.asarray([3,4,5])
    contour_array[1] = np.asarray([300,400,500])
    
    contour_array = fix_contour_orientation(contour_array, axes=axes, verbose=True)
    
    zyx=(500, 4444, 3456)
    zyx = fix_dimension_orientation(zyx, axes = [int(xx) for xx in axes], verbose=True)
    