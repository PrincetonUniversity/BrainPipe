#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:45:19 2017

@author: tpisano


Goal is to register 1p/2p timeseries to 2p volume to LS signal to ABA


To do:
    1) Incorporate resolution of each image
    2) Utilize ability to select region of atlas to register to
Possibly have to use elastix parameter file with 


NEED TO SWITCH TO MHD FORMATS SO TO STORE IMAGE RESOLUTION FORMATS

"""
import os, sys, shutil, tifffile, numpy as np, SimpleITK as sitk
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.io import makedir, listdirfull, change_bitdepth
import subprocess as sp
from tools.registration.masking import generate_masked_atlas
from tools.imageprocessing.orientation import fix_orientation, fix_dimension_orientation
from scipy.ndimage.interpolation import zoom

def multistep_registration(src_lst, dst):
    '''Function to take in a list of tuple pairs where each tuple entry is a dictionary. [(dictionary1, dictionary2), (dictionary2, dictionary3), (...,...)]
    
    Function registers tuple pairs to each other and then registers first entry in first pair with the final entry in last pair.

    Atlas masking is applied BEFORE volume rotation. Only applied if the volume is being used as the fixed image***
    
    ASSUMES Elastix 4.8. Assumes all images are single tiff volumes. Will adjust
    
    
    EVENTUALLY NEED TO ADD IN A WAY TO RESIZE EACH TO FINAL RESOLUTION
    '''
    from tools.registration.register import elastix_command_line_call
    makedir(dst)

    #stepwise registration between pairs
    transform_lst = []    
    for pair in src_lst:
        #try:
            print ('Starting registration of:\n\n{}'.format(pair))
            nm = 'fixed_'+pair[0]['name']+'_with_moving_'+pair[1]['name']
            out = os.path.join(dst, nm); makedir(out); transform_lst.append(out)
            fx = pair[1]['filepath']
            mv = pair[0]['filepath']
            fx_dims = pair[1]['xyz_scale']
            mv_dims = pair[0]['xyz_scale']


            #handle optional masking
            if 'maskatlas' in pair[1]: 
                sys.stdout.write('Masking Atlas.  '); sys.stdout.flush()
                fx_mask = generate_masked_atlas(binarize=True, **{'outputdirectory': dst, 'atlasfile': fx, 'maskatlas': pair[1]['maskatlas']})
                #if 'maskatlas' in pair[0]: mv = generate_masked_atlas(**{'outputdirectory': dst, 'atlasfile': mv, 'maskatlas': pair[1]['maskatlas']})

            #handle volume rotations
            if 'finalorientation' in pair[1]: 
                sys.stdout.write('Rotating fixed image.  '); sys.stdout.flush()
                im = fix_orientation(tifffile.imread(fx), axes=pair[1]['finalorientation'])
                fx = os.path.join(dst, pair[1]['name']+'.tif')
                tifffile.imsave(fx, im); del im
                fx_dims = fix_dimension_orientation(fx_dims, axes=[int(xx) for xx in pair[0]['finalorientation']])
            if 'finalorientation' in pair[0]: 
                sys.stdout.write('Rotating moving image.  '); sys.stdout.flush()
                im = fix_orientation(tifffile.imread(mv), axes=pair[0]['finalorientation'])
                mv = os.path.join(dst, pair[0]['name']+'.tif')
                mv_dims = fix_dimension_orientation(mv_dims, axes=[int(xx) for xx in pair[0]['finalorientation']])
                tifffile.imsave(mv, im); del im

            #convert to MHD
            #################
            mv = convert_to_mhd(mv, mv_dims)
            fx = convert_to_mhd(fx, fx_dims)
            #################
            #run registration
            sys.stdout.write('Starting registration:  \n\n'); sys.stdout.flush()
            
            if 'maskatlas' in pair[1]:#masking
                e_out, transformfile = elastix_command_line_call(fx=fx, mv=mv, out=out, parameters=determine_parameters(pair), fx_mask=fx_mask)
            else:#non masks
                e_out, transformfile = elastix_command_line_call(fx=fx, mv=mv, out=out, parameters=determine_parameters(pair))
            visualize_transform(out)
            generate_registation_visualizations(fx=fx, mv=mv, e_out=e_out, dst=dst, nm=nm, transformfile=transformfile)
        #except:
        #    pass
        
    #apply all transforms to first volume
    apply_multistep_transform(src_lst[0][0]['filepath'], transform_lst)
    
    return

def determine_parameters(pair):
    '''take in files based on parameters to determine which elastix file to use

    Unfinished*****
    '''
    systemdirectory = directorydeterminer()
    import socket
    if socket.gethostname() == 'wanglab-cr8rc42-ubuntu': 
        if any(("maskatlas" in pair[0], "maskatlas" in pair[1])): 
            fls = listdirfull(os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder_local_mask')); fls.sort()
        else:
            fls = listdirfull(os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder_local')); fls.sort()
    
    else:   #need to make the folders for different situations, only necessary for the clusters version currently
    
        if pair[0]['dimensions'] == 2 and pair[0]['dimensions'] == 2:
            fls = os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder')
            
        elif pair[0]['dimensions'] == 2 and pair[0]['dimensions'] == 3:
            fls = os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder')
            
        elif pair[0]['dimensions'] == 3 and pair[0]['dimensions'] == 2:
            fls = os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder')
            
        elif pair[0]['dimensions'] == 3 and pair[0]['dimensions'] == 3:
            fls = os.path.join(systemdirectory, 'wang/pisano/Python/lightsheet/parameterfolder')
    
        fls = listdirfull(fls); fls.sort()
    
    return [xx for xx in fls if '~' not in xx]

def apply_multistep_transform(src, dst, transform_lst):
    '''Take a volume and apply multiple transforms
    
    Inputs
    ----------------
    src = volume to apply transform
    dst = where files will be written
    cores = int # of cores
    transform_lst = list of folders each containing prior output from elastix. In particular TransformParameters.#.txt
        
    '''
    from tools.registration.register import change_transform_parameter_initial_transform
    #generate dst fld
    makedir(dst)
    
    #copy transform parameters
    tick = 0
    for pth in transform_lst:
        for fl in os.listdir(pth):
            if 'TransformParameters' in fl:
                shutil.copy(os.path.join(pth, fl), os.path.join(dst, 'folder'+str(tick).zfill(2)+'_'+fl)) 
        tick+=1
    
    #connect transforms
    tps = [xx for xx in listdirfull(dst) if 'TransformParameters' in xx]; tps.sort(reverse=True) #they are now in reverse order
    for x in range(len(tps)):
        if not x == len(tps)-1:
            change_transform_parameter_initial_transform(tps[x], tps[x+1])

    #run transformix        
    sp.call(['transformix', '-in', src, '-out', dst, '-tp', tps[0]])
            
            
    return

def visualize_transform(src):
    '''Function to visualize iterations of a registration. This requires that the parameter file used has:
        (WriteResultImageAfterEachResolution "true")
    
    Inputs
    ------------
    src: elastix folder
    '''
    #src = '/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/multistep_registration/2p_with_ls_signal'
    #src='/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/multistep_registration/ls_signal_with_ls_auto'
    
    sys.stdout.write('\nConcatenating images to visualize transform...'); sys.stdout.flush()
    
    fls = [xx for xx in listdirfull(src) if '.tif' in xx and '.R' in xx]; fls.sort()
    for xx in range(len(fls)):
        if xx == 0:
            z,y,x = tifffile.imread(fls[xx]).shape
            imm = np.zeros((len(fls), y, x))
        imm[xx,...]=np.max(tifffile.imread(fls[xx]), axis=0)
    
    tifffile.imsave(src+'/visualize_transform.tif', change_bitdepth(imm))
    sys.stdout.write('done\n\n'); sys.stdout.flush()
    return
    

def generate_registation_visualizations(fx, mv, e_out, dst, nm, transformfile):
    '''Function to generate metrics to judge quality of transform
    
    Inputs
    ----------------
    fx = fixed image path
    mv = moving image path
    e_out = path to tif file generated by elastix
    dst = location to save files
    transformfile = final TransformParameters.#.txt file generated by elastix
    
    
    '''
    from tools.imageprocessing.preprocessing import color_movie_merger, resample, combine_images
    
    sys.stdout.write('\nGenerating color movies for registration comparision...'); sys.stdout.flush()
    color_movie_merger(fx, e_out, dst, nm)
    ############################################RGB movie with blue channel=pre-registered stack##########################################
    bluechannel=os.path.join(dst, 'combinedmovies', nm+'_resized_bluechannel.tif')
    resample(mv, fx, svlocname=bluechannel, singletifffile=True) ##needs to be resample(not resample_par) as you need EXTACT dimensions
    color_movie_merger(fx, e_out, dst, nm + '_bluechanneloriginal', movie5=bluechannel)

    ############################################make gridline transform file################################################################
    gridfld, tmpgridline = gridcompare(dst, mv)
    sp.call(['transformix', '-in', tmpgridline, '-out', gridfld, '-tp', transformfile])    
    combine_images(str(mv), fx, os.path.join(gridfld, 'result.tif'), e_out, dst, nm) 
    shutil.rmtree(gridfld)
    
    
    return

def change_resolution(src, in_res, out_res):
    '''Function to change um/pixel resolution
    
    Inputs
    ---------
    src: tiff image path
    in_res: tuple of src's resolution (i.e. (5.0, 5.0, 3.0))
    out_res: desired output tuple of resolution (i.e. (2.5, 2.5, 2.5))
    '''
    
    return zoom(tifffile.imread(src), ([y / x for y,x in zip(in_res, out_res)]))
    
def convert_to_mhd(src, dims, dst=False, verbose = False):
    '''Function to change image from np array(tif) into MHD with appropriate spacing.
    
    Inputs
    ------------------
    src = pth to tiff file
    dims = tuple of um/pixel resolutoin; i.e. (25,25,25)
    dst = (optional) file name and path if requiring different than src+'.mhd'
    
    Returns
    ------------------
    pth to saved mhd file
    '''
    
    im = sitk.GetImageFromArray(tifffile.imread(src))
    #print im
    im.SetSpacing(dims)
    if src[-4:] == '.tif': src=src[:-4]
    
    if dst: 
        if dst[-4:] != '.mhd': dst=dst+'.mhd'
        sitk.WriteImage(im, dst)
        if verbose: print(dst)
        return dst
    else:
        sitk.WriteImage(im, src+'.mhd')
        if verbose: print(src+'.mhd')
        return src+'.mhd'

    
def gridcompare(dst, mv): 
    gridfld=os.path.join(dst, 'tmpgridlinefilefortransformix')
    makedir(gridfld)
    tmpgridline=os.path.join(gridfld, 'gridline.tif')
    gridlineresized=gridlineresizer(mv)
    tifffile.imsave(tmpgridline, gridlineresized)
    print ('gridline resized to shape {}'.format(gridlineresized.shape)) 
    return gridfld, tmpgridline 
    
    
def gridlineresizer(mv, gridlinefile=None):
    '''resizes gridline file to appropraite z dimenisions.
    Returns path to zcorrected stack
    Input:
       movingfile: atlas file used to scale gridlinefile, can be path, np object, or volume class
       savlocation: destination of file'''
    import SimpleITK as sitk, cv2
    ###determine output image stack size   
    if (type(mv).__module__ == np.__name__) == True:
        z,y,x=mv.shape            
    elif isinstance(mv, str):
        mv=sitk.GetArrayFromImage(sitk.ReadImage(mv))
        z,y,x=mv.shape
   ##load gridline file
    if not gridlinefile: gridlinefile = 'home/wanglab/wang/pisano/Python/lightsheet/supp_files/gridlines.tif'
    grdfl=tifffile.imread(gridlinefile)
    if len(grdfl.shape)==2:
        grdim=grdfl
    else:
        grdim = grdfl[0,:,:]
   ##########preprocess gridline file: Correcting for z difference between gridline file and atlas   
    grdresized=cv2.resize(grdim, (x,y), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
    grdresizedstck=np.zeros((z, y, x))
    for i in range(z):
        grdresizedstck[i,:,:]=grdresized
    return change_bitdepth(grdresizedstck)


if __name__ == '__main__':
    #inputs
    systemdirectory = directorydeterminer()
    dst = os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/multistep_registration')


    one_p={
    'name': '1p',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'labeltype': 'doric', 
    'filepath': os.path.join(systemdirectory, 'wang/pisano/cfos_experiment/tif/tif/0000025335_an10_cropped.tif'), #<--need to adjust this from Color to gray
    'dimensions': 2,
    #'xyz_scale': (3.75,3.75,3.75), #need to implement this; #approximately 800pixels in 3mm therefore ~3.75um/pixel
    #'maskatlas': {'x': all, 'y': '125:202', 'z': '75:125'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }

    two_p={
    'name': '2p',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'labeltype': '2p', 
    'filepath': os.path.join(systemdirectory, 'wang/mkislin/2p-wf/20170117_L7Cre-MK122016_X_ArchT-gfp/place01-2-4.tif'),
    'dimensions': 3,
    'xyz_scale': (1.65,1.65,5), #um/pixel; need to implement this
    #'maskatlas': {'x': all, 'y': '125:202', 'z': '75:125'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #LEFT IS UP ON THIS ATLAS CURRENTLY
    'finalorientation' :  ('-1','0','-2'),#'finalorientation' :  ('-1','0','-2'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }

    two_p_cropped={
    'name': '2p_cropped',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'labeltype': '2p', 
    'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/multistep_registration/2p_cropped.tif'),
    'dimensions': 3,
    'xyz_scale': (1.65,1.65,5), #um/pixel; need to implement this
    #'maskatlas': {'x': all, 'y': '125:202', 'z': '75:125'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #LEFT IS UP ON THIS ATLAS CURRENTLY
    #'finalorientation' :  ('-1','0','-2'),#'finalorientation' :  ('-1','0','-2'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }
    
    
    ls_signal_5d2x={
    'name': 'ls_5d2x_signal',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/mk10_cb_5d2x/201701_mk10_5d2x_488_647_0010na_1hfss_z3um_10povlap_1000msec_resized_ch01_resampledforelastix.tif'),
    #'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/201701_cfos_mk10/201701_cfos_mk10_488_555_647_0010na_1hfsds_z3um_250msec_resized_ch02_resampledforelastix.tif'),
    'dimensions': 3,
    'xyz_scale': (21.5, 7.5, 11.2), #um/pixel; need to implement this
    #'maskatlas': {'x': '370:1200', 'y': '600:1200', 'z': all}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    'maskatlas': {'x': '100:350', 'y': '90:325', 'z': '200:592'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }
    
    ls_signal={
    'name': 'ls_signal',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/201701_cfos_mk10/201701_cfos_mk10_488_555_647_0010na_1hfsds_z3um_250msec_resized_ch02.tif'),
    #'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/201701_cfos_mk10/201701_cfos_mk10_488_555_647_0010na_1hfsds_z3um_250msec_resized_ch02_resampledforelastix.tif'),
    'dimensions': 3,
    'xyz_scale': (1.8, 15.4,15.8), #um/pixel; need to implement this
    #'xyz_scale': (19.2, 19.2,19.2), #um/pixel; need to implement this
    'maskatlas': {'x': '370:1200', 'y': '600:1200', 'z': all}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'maskatlas': {'x': '50:300', 'y': '515:700', 'z': all}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }

    ls_auto={
    'name': 'ls_auto',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'filepath': os.path.join(systemdirectory, 'wang/pisano/tracing_output/cfos/201701_cfos/201701_cfos_mk10/201701_cfos_mk10_488_555_647_0010na_1hfsds_z3um_250msec_resized_ch00_resampledforelastix.tif'),
    'dimensions': 3,
    'xyz_scale': (19.2, 19.2,19.2), #um/pixel; need to implement this
    #'maskatlas': {'x': all, 'y': '125:202', 'z': '75:125'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }

    aba={
    'name': 'allen_brain_atlas',
    'systemdirectory':  directorydeterminer(), #don't need to touch
    'filepath': os.path.join(systemdirectory, 'wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'),
    'dimensions': 3,
    'xyz_scale': (25,25,25), #um/pixel; need to implement this
    #'maskatlas': {'x': all, 'y': '125:202', 'z': '75:125'}, #dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out.        
    #'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation
    }

    #src_lst = [[one_p, two_p], [two_p, ls_signal], [ls_signal, ls_auto], [ls_auto, aba]]
    src_lst = [[one_p, two_p], [two_p, ls_signal_5d2x], [ls_signal, ls_auto], [ls_auto, aba]]
    src_lst=src_lst[1:]
    #multistep_registration(src_lst, dst)    
    from tools.registration.multistep_registration import multistep_registration

    multistep_registration([[two_p_cropped, ls_signal_5d2x]], dst)
