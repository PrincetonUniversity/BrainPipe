#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:48 2019

@author: wanglab
"""

import SimpleITK as sitk
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    # range to clip y
    yrng = (76, 822)
    xrng = (20, 475)
    zrng = (62, 364)

    # load in wh colored atlas and reorient
    ann = np.fliplr(sitk.GetArrayFromImage(sitk.ReadImage(
        "/jukebox/LightSheetData/brodyatlas/atlas/original/WHS_SD_rat_atlas_v3.nii")))
    nann = "/jukebox/LightSheetData/brodyatlas/atlas/modified/WHS_SD_rat_atlas_v3_anterior_up_DV.tif"
    # here we are reorienting to A-P, D-V orientation (how our images are taken) and THEN CROP
    ann = ann[::-1][zrng[0]:zrng[1], yrng[0]:yrng[1], xrng[0]:xrng[1]]
    tifffile.imsave(nann, ann)

    auto = np.fliplr(sitk.GetArrayFromImage(sitk.ReadImage(
        "/jukebox/LightSheetData/brodyatlas/atlas/original/WHS_SD_rat_T2star_v1.01.nii")))
    nauto = "/jukebox/LightSheetData/brodyatlas/atlas/modified/WHS_SD_rat_T2star_v1.01_anterior_up_DV.tif"
    auto = auto[::-1][zrng[0]:zrng[1], yrng[0]:yrng[1], xrng[0]:xrng[1]]
    tifffile.imsave(nauto, auto)

    # remove skull in atlas using mask
    from skimage.external import tifffile
    mask = tifffile.imread(
        "/jukebox/LightSheetData/brodyatlas/atlas/modified/WHS_SD_rat_atlas_v3_anterior_up_DV.tif")
    atlas = tifffile.imread(
        "/jukebox/LightSheetData/brodyatlas/atlas/modified/WHS_SD_rat_T2star_v1.01_anterior_up_DV.tif")
    atlas[mask == 0] = 0
    tifffile.imsave(
        "/jukebox/LightSheetData/brodyatlas/atlas/modified/WHS_SD_rat_T2star_v1.01_anterior_up_skullremoved_DV.tif", atlas)

    # make df for labels
    ndf = pd.DataFrame(columns=["name", "id"])

    # wh labels - index = pix value
    wh = "/jukebox/LightSheetData/brodyatlas/atlas/original/WHS_SD_rat_atlas_v3.label"
    with open(wh, "r") as w:
        lines = w.readlines()
        w.close()

    # generate dataframe
    lines = lines[14:]
    for i in range(len(lines)):
        print("{} of {}".format(i, len(lines)))
        line = lines[i]
        vals, structure, empty = line.split(""")
        idx, r, g, b, a, vis, msh = vals.split(); r=int(r); g=int(g); b=int(b)
        ndf.loc[i] = [structure, idx]

    ndf.to_csv("/jukebox/LightSheetData/brodyatlas/atlas/modified/labels_v3.csv")
