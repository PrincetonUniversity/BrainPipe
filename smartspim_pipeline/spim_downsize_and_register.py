#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:04:02 2020

@author: wanglab
"""

import os
import numpy as np
import tifffile as tif
import SimpleITK as sitk
import cv2
import multiprocessing as mp
import sys
from scipy.ndimage import zoom
sys.path.append("/scratch/ejdennis/rat_BrainPipe")
from tools.registration.register import elastix_command_line_call


def fast_scandir(dirname):
    """ gets all folders recursively """
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


def get_folderstructure(dirname):
    folderstructure = []
    for i in os.walk(src):
        folderstructure.append(i)
    return folderstructure


def downsize_folder_of_tiffs(pth, dst, atlpth):
    print("\ndownsizing def with dst is %s\n" % dst)
    print(" pth is %s " % pth)
    imgs = [os.path.join(pth, xx) for xx in os.listdir(pth) if "tif" in xx]
    z = len(imgs)
    resizef = 5  # factor to downsize imgs by
    iterlst = [(img, dst, resizef) for img in imgs]
    p = mp.Pool(11)
    p.starmap(resize_helper, iterlst)
    p.terminate()

    # now downsample to 140% of pra atlas
    imgs = [os.path.join(dst, xx) for xx in os.listdir(dst) if "tif" in xx]
    imgs.sort()
    z = len(imgs)
    y, x = sitk.GetArrayFromImage(sitk.ReadImage(imgs[0])).shape
    arr = np.zeros((z, y, x))
    atl = sitk.GetArrayFromImage(sitk.ReadImage(atlpth))
    atlz, atly, atlx = atl.shape  # get shape, sagittal
    # read all the downsized images
    for i, img in enumerate(imgs):
        if i % 5000 == 0:
            print(i)
        arr[i, :, :] = sitk.GetArrayFromImage(sitk.ReadImage(img))
    # switch to sagittal
    arrsag = np.swapaxes(arr, 2, 0)
    z, y, x = arrsag.shape
    print((z, y, x))
    print("\n**********downsizing....heavy!**********\n")

    arrsagd = zoom(arrsag, ((atlz*1.4/z), (atly*1.4/y), (atlx*1.4/x)), order=1)
    tif.imsave(os.path.join(os.path.dirname(dst), "downsized_for_atlas.tif"),
               arrsagd.astype("uint16"))


def register_ch(cell,mv,fx, param_fld, svpth):
        out = os.path.join(svpth, "elastix")
        if not os.path.exists(out):
            os.mkdir(out)

        params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
        # run
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

        if cell:
            out = os.path.join(svpth, "elastix/cell_to_reg")
            if not os.path.exists(out):
                os.mkdir(out)

            params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
            # run
            e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)


def resize_helper(img, dst, resizef):
    print(os.path.basename(img))
    im = sitk.GetArrayFromImage(sitk.ReadImage(img))
    y, x = im.shape
    yr = int(y/resizef)
    xr = int(x/resizef)
    im = cv2.resize(im, (xr, yr), interpolation=cv2.INTER_LINEAR)
    tif.imsave(os.path.join(dst, os.path.basename(img)),
               im.astype("uint16"), compress=1)


if __name__ == "__main__":

    # takes 2 command line args
    print(sys.argv)

    src = sys.argv[1]  # main image folder
    svpth = sys.argv[2]  # save path
    atlpth = "/jukebox/brody/ejdennis/lightsheet/PRA_25um.tif"  # rat_atlas
    param_fld = "/scratch/ejdennis/rat_registration_parameter_folder"


    # NOTE!!! THIS CHANGES IMAGES FROM HORIZ TO SAGITTAL!!

    # make sure the save path exists, if not, make it
    if not os.path.exists(svpth):
        os.mkdir(svpth)
    if not os.path.exists(os.path.join(svpth,"reg_ch")):
        os.mkdir(os.path.join(svpth,"reg_ch"))
    if not os.path.exists(os.path.join(svpth,"cell_ch")):
        os.mkdir(os.path.join(svpth,"cell_ch"))
    # get relevant foldernames from the structure
    for directory, subdirectories, files in get_folderstructure(src):
        if "rawdata" in directory:
            for ls in subdirectories:
                if 'resolution' in ls:
                    rawdata = os.path.join(directory, ls)

    for i in os.listdir(rawdata):
        if '488' in i:
            for j in os.listdir(os.path.join(rawdata, i)):
                if 'corrected' in j:
                    reg_ch = fast_scandir(os.path.join(rawdata, i, j))[-1]
        if '64' in i:
            for j in os.listdir(os.path.join(rawdata, i)):
                if 'corrected' in j:
                    cell_ch = fast_scandir(os.path.join(rawdata, i, j))[-1]

    print("\nrawdata folder is: %s\n" % rawdata)
    # path to store downsized images

    print("\ndownsizing %s \n" % os.path.join(svpth,"reg_ch"))
    downsize_folder_of_tiffs(reg_ch, os.path.join(svpth,"reg_ch"), atlpth)

    print("\ndownsizing %s \n" % os.path.join(svpth,"cell_ch"))
    downsize_folder_of_tiffs(cell_ch, os.path.join(svpth,"cell_ch"), atlpth)

    print("\n probably finished downsizing successfully \n")

    print("\nregistering %s to %s" % (os.path.join(svpth,"reg_ch","downsized_for_atlas.tif"),
    atlpth))
    register_ch(0,
                os.path.join(svpth,"reg_ch","downsized_for_atlas.tif"),
                atlpth,
                param_fld,
                svpth)
    print("\n probably finished registering reg_ch to atlas successfully \n")

    print("\nregistering %s to %s" % (os.path.join(svpth,"cell_ch","downsized_for_atlas.tif"),
    os.path.join(svpth,"reg_ch","downsized_for_atlas.tif")))
    register_ch(1,
                os.path.join(svpth,"cell_ch","downsized_for_atlas.tif"),
                os.path.join(svpth,"reg_ch","downsized_for_atlas.tif"),
                param_fld,
                svpth)
    print("\n probably finished registering cell_ch to reg_ch successfully \n")

    print("\ninverse registering %s to %s" % (os.path.join(svpth,"reg_ch","downsized_for_atlas.tif"),
    atlpth))

    register_ch(0,
                atlpth,
                os.path.join(svpth,"reg_ch","downsized_for_atlas.tif"),
                param_fld,
                svpth)
    print("\n probably finished registering atlas to reg_ch successfully \n")
