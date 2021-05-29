
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:39:28 2020
@author: wanglab
"""

import tifffile as tif
import shutil
import numpy as np
import sys
import os
import pickle
import SimpleITK as sitk
from scipy.io import loadmat, savemat

def transform_points(src, dst, transformfiles, resample_points=False,reorient=False):
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
    # load
    cells = np.load(src)

    # optionally resample points
    if resample_points:
        original_dims, resample_dims = resample_points
        cells = points_resample(cells, original_dims, resample_dims, reorient)
    print('resampled')
    # generate text file
    pretransform_text_file = create_text_file_for_elastix(cells, dst)
    print('made txt')
    # copy over elastix files
    transformfiles = modify_transform_files(transformfiles, dst)
    print('modified transform files')
    # run transformix on points
    points_file = point_transformix(pretransform_text_file, transformfiles[-1], dst)
    print('ran transformix')
    # convert registered points into structure counts
    unpack_pnts(points_file, dst)

    return


def create_text_file_for_elastix(src, dst):
    """

    Inputs
    ---------
    src = numpy file consiting of nx3 (ZYX points)
    dst = folder location to write points
    """

    print("This function assumes ZYX centers...")

    # setup folder
    if not os.path.exists(dst):
        os.mkdir(dst)

    # create txt file, with elastix header, then populate points
    pth = os.path.join(dst, "zyx_points_pretransform.txt")

    # load
    if type(src) == np.ndarray:
        arr = src
    else:
        arr = np.load(src) if src[-3:] == "npy" else loadmat(src)["cell_centers_orig_coord"]

    # convert
    stringtowrite = "\n".join(["\n".join(["{} {} {}".format(i[2], i[1], i[0])])
                               for i in arr])  # this step converts from zyx to xyz*****

    # write file
    sys.stdout.write("writing centers to transfomix input points text file...")
    sys.stdout.flush()
    with open(pth, "w+") as fl:
        fl.write("index\n{}\n".format(len(arr)))
        fl.write(stringtowrite)
        fl.close()
    sys.stdout.write("...done writing centers\n")
    sys.stdout.flush()

    return pth


def modify_transform_files(transformfiles, dst):
    """Function to copy over transform files, modify paths in case registration was done on the cluster, and tether them together

        Inputs
    ---------
    transformfiles =
        list of all elastix transform files used, and in order of the original transform****

    """

    # new
    ntransformfiles = [os.path.join(dst, "order{}_{}".format(
        i, os.path.basename(xx))) for i, xx in enumerate(transformfiles)]

    # copy files over
    [shutil.copy(xx, ntransformfiles[i]) for i, xx in enumerate(transformfiles)]

    # modify each with the path
    for i, pth in enumerate(ntransformfiles):

        # skip first
        if i != 0:

            # read
            with open(pth, "r") as fl:
                lines = fl.readlines()
                fl.close()

            # copy
            nlines = lines

            # iterate
            for ii, line in enumerate(lines):
                if "(InitialTransformParametersFileName" in line:
                    nlines[ii] = "(InitialTransformParametersFileName {})\n".format(
                        ntransformfiles[i-1])

            # write
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
    # set paths
    trnsfrm_out_file = os.path.join(dst, "outputpoints.txt")

    # run transformix point transform
    call = "transformix -def {} -out {} -tp {}".format(pretransform_text_file, dst, transformfile)
    print(check_output(call, shell=True))
    sys.stdout.write("\n   Transformix File Generated: {}".format(trnsfrm_out_file))
    sys.stdout.flush()
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

    # inputs
    assert type(points_file) == str
    point_or_index = 'OutputPoint'

    # get points
    with open(points_file, "r") as f:
        lines = f.readlines()
        f.close()

    # populate post-transformed array of contour centers
    sys.stdout.write("\n\n{} points detected\n\n".format(len(lines)))
    arr = np.empty((len(lines), 3))
    for i in range(len(lines)):
        arr[i, ...] = lines[i].split()[lines[i].split().index(point_or_index) +
                                       3:lines[i].split().index(point_or_index)+6]  # x,y,z

    # optional save out of points
    dst_fl = os.path.join(dst, "posttransformed_zyx_voxels.npy")
    np.save(dst_fl, np.asarray([(z, y, x) for x, y, z in arr]))

    # check to see if any points where found
    print("output array shape {}".format(arr.shape))

    return dst_fl


def points_resample(src, original_dims, resample_dims, points_resample, verbose=True):
    """Function to adjust points given resizing by generating a transform matrix

    ***Assumes ZYX and that any orientation changes have already been done.***

    src: numpy array or list of np arrays of dims nx3
    original_dims (tuple)
    resample_dims (tuple)
    """

    src = np.asarray(src)
    if src.shape[-1] > 3:
        if points_resample:
            newsrc = []
            for a in src:
                newsrc.append([list(a)[2],list(a)[1],list(a)[0]])
            src = np.asarray(newsrc)
        else:
            newsrc=[]
            for a in src:
                newsrc.append(list(a)[0:3])
            src=np.asarray(newsrc)
    assert src.shape[-1]==3; "src needs to be nd3"

    # initialize
    d1, d2 = src.shape
    nx4centers = np.ones((d1, d2+1))
    nx4centers[:, :-1] = src

    # acount for resampling by creating transformmatrix
    zr, yr, xr = resample_dims
    z, y, x = original_dims

    # apply scale diff
    trnsfrmmatrix = np.identity(4)*(zr/float(z), yr/float(y), xr/float(x), 1)
    if verbose:
        sys.stdout.write("trnsfrmmatrix:\n{}\n".format(trnsfrmmatrix))

    # nx4 * 4x4 to give transform
    trnsfmdpnts = nx4centers.dot(trnsfrmmatrix)  # z,y,x
    if verbose:
        sys.stdout.write("first three transformed pnts:\n{}\n".format(trnsfmdpnts[0:3]))

    return trnsfmdpnts


def load_dictionary(pth):
    """simple function to load dictionary given a pth
    """
    kwargs = {}
    with open(pth, "rb") as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs

def get_dims(input_path):
    """ gets resize_dims values """
    imgs=[]
    for path, subdirs, files in os.walk(input_path):
        for name in files:
            if "tif" in name:
                imgs.append(os.path.join(path,name))

    z = len(imgs)
    y,x = sitk.GetArrayFromImage(sitk.ReadImage(imgs[1])).shape
    zyx_tuple = (z,y,x)
    return zyx_tuple


# %%
if __name__ == "__main__":

    file_dict = {"j317_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-001/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/", 
             "j316_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-002/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "e153_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-003/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "e142_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-004/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "h234_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-005/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "j319_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-006/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "a253_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-007/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "e143_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-008/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "e144_488": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-009/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/",
             "j317_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-001/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/", 
             "j316_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-002/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "e153_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-003/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "e142_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-004/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "h234_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-005/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "j319_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-006/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "a253_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-007/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "e143_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-008/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/",
             "e144_642": "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-009/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected/"}

    # NOTE CHECK TO ENSURE ACCOUNTING FOR INPUT RESAMPLING, and ORIENTATION CHANGE*****
    # inputs
    # numpy file consiting of nx3 (ZYX points) or if .mat file structure where zyx is called "cell_centers_orig_coord"
    # folder location to write points
    dst_fld = "/scratch/ejdennis/spim_cell_centers/transformed_cells"
    if not os.path.exists(dst_fld):
        os.mkdir(dst_fld)    

    fld = "/scratch/ejdennis/spim_cell_centers"
    tfld = "/scratch/ejdennis/spim_transforms"
    resample_dims = (866,539,1610) #140% of mPRA
    mPRA_dims = (618,385,1150)
    fld_list = [xx for xx in os.listdir(fld) if "npy" in xx]
    print("fld list: {}".format(fld_list))
    for subsrc in fld_list:
        src= os.path.join(fld,subsrc) #path to cell centers
        nm = subsrc[0:4]
        print("name is {}".format(nm))
        resample_dims = (866,539,1610) #140% of mPRA
        
        if "488" in subsrc:
            transform_inverse = os.path.join(tfld,nm,"elastix_inverse_transform")
            file_key = "{}_488".format(nm)
            raw_folder_name = file_dict[file_key]
            print("488 raw folder name".format(raw_folder_name))
            original_dims = get_dims(raw_folder_name)
            flip=0
            dst_fld = "/scratch/ejdennis/spim_cell_centers/transformed_cells"
            if not os.path.exists(dst_fld):
                os.mkdir(dst_fld)
        elif "642" in subsrc:
            transform_inverse_reg = os.path.join(tfld,nm,"reg_to_cell")
            raw_folder_name = file_dict["{}_642".format(nm)]
       	    print("642 raw folder name".format(raw_folder_name))
            transformfiles = [os.path.join(transform_inverse_reg,"TransformParameters.0.txt")]
            original_dims = get_dims(raw_folder_name)
            resample_points = [original_dims,resample_dims]
            dst = os.path.join(dst_fld,"{}_642_in_488".format(nm))
            if not os.path.exists(dst):
                os.mkdir(dst)
            transform_points(src,dst,transformfiles,resample_points)
            src = os.path.join(dst,"posttransformed_zyx_voxels.npy")
            original_dims = resample_dims
       	    transform_inverse = os.path.join(tfld,nm,"elastix_inverse_transform")        
            flip = 1
            dst= os.path.join(dst_fld,"{}_642_in_atl".format(nm))
            if not os.path.exists(dst):
                os.mkdir(dst)
        resample_points = [original_dims, resample_dims]

        transformfiles = [os.path.join(transform_inverse,"TransformParameters.0.txt"),
            os.path.join(transform_inverse,"TransformParameters.1.txt"),
            os.path.join(transform_inverse,"TransformParameters.2.txt"),
            os.path.join(transform_inverse,"TransformParameters.3.txt")]

        # apply
        transform_points(src, dst, transformfiles, resample_points,flip)

