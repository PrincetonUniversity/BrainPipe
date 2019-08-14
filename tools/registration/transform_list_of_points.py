#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:47:47 2018

@author: tpisano

Set of functions to take a list of points and transform them into new space

# NEED ELASTIX TO RUN THIS (so run on a LINUX machine)

"""

import os, sys, numpy as np, shutil
from scipy.io import loadmat
import pickle

def transform_points(src, dst, transformfiles, resample_points=False, param_dictionary_for_reorientation=False):
    """
    
    Inputs
    ---------
    src = numpy file consiting of nx3 (ZYX points)
    dst = folder location to write points
    transformfiles = 
        list of all elastix transform files used, and in order of the original transform****
    resample_points = [original_dims, resample_dims] if there was resampling done, use this here
    param_dictionary_for_reorientation = param_dictionary for lightsheet package to use for reorientation
    """
    #load
    src = np.load(src) if src[-3:] == "npy" else loadmat(src)["cell_centers_orig_coord"]
    
    #reorient - test to make sure this works
    if param_dictionary_for_reorientation:
        from tools.imageprocessing.orientation import fix_contour_orientation
        kwargs = load_dictionary(param_dictionary_for_reorientation)
        if resample_points:
            from tools.imageprocessing.orientation import fix_dimension_orientation
            resample_points = [fix_dimension_orientation(resample_points[0], **kwargs), resample_points[1]] #fix dims
        src = fix_contour_orientation(src, axes=None, verbose = True, **kwargs)
            
    #optionally resample points
    if resample_points:
        original_dims, resample_dims = resample_points
        src = points_resample(src, original_dims, resample_dims)
    
    #generate text file
    pretransform_text_file = create_text_file_for_elastix(src, dst)
    
    #copy over elastix files
    transformfiles = modify_transform_files(transformfiles, dst)
   
    #run transformix on points
    points_file = point_transformix(pretransform_text_file, transformfiles[-1], dst)
    
    #convert registered points into structure counts
    unpack_pnts(points_file, dst)   
    
    return
    
def makedir(dst):
    if not os.path.exists(dst):os.mkdir(dst)
    return
    
def create_text_file_for_elastix(src, dst):
    """
    
    Inputs
    ---------
    src = numpy file consiting of nx3 (ZYX points)
    dst = folder location to write points
    """
    
    print("This function assumes ZYX centers...")
    
    #setup folder
    makedir(dst)
                                 
    #create txt file, with elastix header, then populate points
    pth=os.path.join(dst, "zyx_points_pretransform.txt")
    
    #load
    if type(src) == np.ndarray:
        arr = src
    else:
        arr = np.load(src) if src[-3:] == "npy" else loadmat(src)["cell_centers_orig_coord"]
    
    #convert
    stringtowrite = "\n".join(["\n".join(["{} {} {}".format(i[2], i[1], i[0])]) for i in arr]) ####this step converts from zyx to xyz*****
    
    #write file
    sys.stdout.write("writing centers to transfomix input points text file..."); sys.stdout.flush()
    with open(pth, "w+") as fl:
        fl.write("index\n{}\n".format(len(arr)))    
        fl.write(stringtowrite)
        fl.close()
    sys.stdout.write("...done writing centers\n"); sys.stdout.flush()
        
    return pth

def modify_transform_files(transformfiles, dst):
    """Function to copy over transform files, modify paths in case registration was done on the cluster, and tether them together
    
        Inputs
    ---------
    transformfiles = 
        list of all elastix transform files used, and in order of the original transform****
    
    """
    
    #new
    ntransformfiles = [os.path.join(dst, "order{}_{}".format(i,os.path.basename(xx))) for i,xx in enumerate(transformfiles)]
    
    #copy files over
    [shutil.copy(xx, ntransformfiles[i]) for i,xx in enumerate(transformfiles)]
    
    #modify each with the path
    for i,pth in enumerate(ntransformfiles):
        
        #skip first
        if i!=0:
            
            #read
            with open(pth, "r") as fl:
                lines = fl.readlines()
                fl.close()
            
            #copy
            nlines = lines
            
            #iterate
            for ii, line in enumerate(lines):
                if "(InitialTransformParametersFileName" in line:
                    nlines[ii] = "(InitialTransformParametersFileName {})\n".format(ntransformfiles[i-1])
            
            #write
            with open(pth, "w") as fl:
                for nline in lines:
                    fl.write(str(nline))
                fl.close()
        
    return ntransformfiles
    
    

def point_transformix(pretransform_text_file, transformfile, dst):
    """apply elastix transform to points
    
    
    Inputs
    -------------
    pretransform_text_file = list of points that already have resizing transform
    transformfile = elastix transform file
    dst = folder
    
    Returns
    ---------------
    trnsfrm_out_file = pth to file containing post transformix points
    
    """
    sys.stdout.write("\n***********Starting Transformix***********")
    from subprocess import check_output
    #set paths    
    trnsfrm_out_file = os.path.join(dst, "outputpoints.txt")
    
    #run transformix point transform
    call = "transformix -def {} -out {} -tp {}".format(pretransform_text_file, dst, transformfile)
    print(check_output(call, shell=True))
    sys.stdout.write("\n   Transformix File Generated: {}".format(trnsfrm_out_file)); sys.stdout.flush()
    return trnsfrm_out_file


def unpack_pnts(points_file, dst):
    """
    function to take elastix point transform file and return anatomical locations of those points
    
    Here elastix uses the xyz convention rather than the zyx numpy convention
    
    Inputs
    -----------
    points_file = post_transformed file, XYZ
    
    Returns
    -----------
    dst_fl = path to numpy array, ZYX
    
    """   

    #####inputs 
    assert type(points_file)==str
    point_or_index = 'OutputPoint'
    
    #get points
    with open(points_file, "r") as f:                
        lines=f.readlines()
        f.close()

    #####populate post-transformed array of contour centers
    sys.stdout.write("\n\n{} points detected\n\n".format(len(lines)))
    arr=np.empty((len(lines), 3))    
    for i in range(len(lines)):        
        arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            
    #optional save out of points
    dst_fl = os.path.join(dst, "posttransformed_zyx_voxels.npy")
    np.save(dst_fl, np.asarray([(z,y,x) for x,y,z in arr]))
    
    #check to see if any points where found
    print("output array shape {}".format(arr.shape))
        
    return dst_fl

def points_resample(src, original_dims, resample_dims, verbose = False):
    """Function to adjust points given resizing by generating a transform matrix
    
    ***Assumes ZYX and that any orientation changes have already been done.***
    
    src: numpy array or list of np arrays of dims nx3
    original_dims (tuple)
    resample_dims (tuple)
    """
    src = np.asarray(src)
    assert src.shape[-1] == 3, "src must be a nx3 array"
    
    #initialize
    d1,d2=src.shape
    nx4centers=np.ones((d1,d2+1))
    nx4centers[:,:-1]=src
    
    #acount for resampling by creating transformmatrix
    zr, yr, xr = resample_dims
    z, y, x = original_dims
    
    #apply scale diff
    trnsfrmmatrix=np.identity(4)*(zr/float(z), yr/float(y), xr/float(x), 1)
    if verbose: sys.stdout.write("trnsfrmmatrix:\n{}\n".format(trnsfrmmatrix))
    
    #nx4 * 4x4 to give transform
    trnsfmdpnts=nx4centers.dot(trnsfrmmatrix) ##z,y,x
    if verbose: sys.stdout.write("first three transformed pnts:\n{}\n".format(trnsfmdpnts[0:3]))

    return trnsfmdpnts


def load_dictionary(pth):
    """simple function to load dictionary given a pth
    """
    kwargs = {};
    with open(pth, "rb") as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs

if __name__ == "__main__":
    
    
    ###NOTE CHECK TO ENSURE ACCOUNTING FOR INPUT RESAMPLING, and ORIENTATION CHANGE*****
    
    from tools.registration.transform_list_of_points import *
    
    #inputs
    #numpy file consiting of nx3 (ZYX points) or if .mat file structure where zyx is called "cell_centers_orig_coord"
    src = "/home/wanglab/Downloads/transform_test/nx3_zyx_points.npy"
    dst = "/home/wanglab/Downloads/transform_test"; makedir(dst) # folder location to write points
    
    #dummy array
    np.save(src, (np.random.rand(10,3)*20).astype("int"))
    
    #EXAMPLE USING CLEARMAP - when marking centers in the  "raw" full sized cfos channel. This will transform those centers into "atlas" space (in this case the moving image)
    #list of all elastix transform files used, and in order of the original transform****
    transformfiles = ["/jukebox/wang/seagravesk/lightsheet/201710_cfos/f37107_demons/clearmap_cluster_output/elastix_auto_to_atlas/TransformParameters.0.txt", # this is auto = fixed image; atlas = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos/f37107_demons/clearmap_cluster_output/elastix_auto_to_atlas/TransformParameters.1.txt", # this is auto = fixed image; atlas = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos/f37107_demons/clearmap_cluster_output/elastix_cfos_to_auto/TransformParameters.0.txt",  # this is cfos = fixed image; auto = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos/f37107_demons/clearmap_cluster_output/elastix_cfos_to_auto/TransformParameters.1.txt"]  # this is cfos = fixed image; auto = moving image
    
    #apply - resample_points not accounted for yet in cfos
    transform_points(src, dst, transformfiles)
    
    #EXAMPLE USING LIGHTSHEET - when marking centers in the  "raw" full sized cfos channel. This will transform those centers into "atlas" space (in this case the moving image)
    # in this case the "inverse transform has the atlas as the moving image in the first step, and the autofluorescence channel as the moving image in the second step 
    transformfiles = ["/jukebox/wang/seagravesk/lightsheet/201710_cfos_left_side_only_registration/m37071_demons/elastix_inverse_transform/cellch_m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec/m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec_resized_ch00_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.0.txt", # this is auto = fixed image; atlas = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos_left_side_only_registration/m37071_demons/elastix_inverse_transform/cellch_m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec/m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec_resized_ch00_resampledforelastix_atlas2reg2sig/atlas2reg_TransformParameters.1.txt", # this is auto = fixed image; atlas = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos_left_side_only_registration/m37071_demons/elastix_inverse_transform/cellch_m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec/m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec_resized_ch00_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.0.txt", # this is cfos = fixed image; auto = moving image
                      "/jukebox/wang/seagravesk/lightsheet/201710_cfos_left_side_only_registration/m37071_demons/elastix_inverse_transform/cellch_m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec/m37071_demonstrator_20171006_790_015na_1hfsds_z5um_1000msec_resized_ch00_resampledforelastix_atlas2reg2sig/reg2sig_TransformParameters.1.txt"] # this is cfos = fixed image; auto = moving image   

    #optional resampling between fullsized and input to elastix
    original_dims = (1454, 2560, 2160)
    resample_dims = (592, 686, 416)
    resample_points = [original_dims, resample_dims]
    
    #if there was an orientation change
    param_dictionary_for_reorientation = "/jukebox/wang/seagravesk/lightsheet/201710_cfos_left_side_only_registration/m37071_demons/param_dict.p"
    
    #apply
    transform_points(src, dst, transformfiles, resample_points)


    # Code for converting .npy to .mat
    f = "/some/path/.npy" # full path to the .npy that you want converted
    new_path = "some_other_path.mat" # full path to where the .mat file should be saved and what it should be named
    data = np.load(f) # Load the .npy matrix into the workspace
    save_dict = dict(x=data) # convert it to a dictionary
    savemat(new_path, save_dict) # save the dictionary as a .mat file

