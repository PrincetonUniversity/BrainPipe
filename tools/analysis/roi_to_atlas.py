#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:51:13 2019

@author: tpisano
"""
from tools.registration.transform import transformed_pnts_to_allen_helper_func
from tools.registration.register import elastix_command_line_call, jacobian_command_line_call, change_interpolation_order, transformix_command_line_call, count_structure_lister
from tools.conv_net.input.read_roi import read_roi_zip
import os
import sys
import shutil
import numpy as np
import tifffile

# sys.path.append("/jukebox/wang/pisano/Python/lightsheet")

if __name__ == "__main__":
    # load in ROIS - clicked in horizontal volume
    roi_pth =
    # "/jukebox/wang/willmore/lightsheet/20181213_fiber_optic_placement/processed/m278/20190211_fiber_points_horizontal_RoiSet.zip"
    zyx_rois = np.asarray([[int(yy) for yy in xx.replace(".roi", "").split("-")]
                           for xx in read_roi_zip(roi_pth, include_roi_name=True)])

    # go from horiztonal to sag
    zyx_rois = np.asarray([[xx[2], xx[1], xx[0]] for xx in zyx_rois])

    # convert to structure
    annotation_file = "/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annotation.tif"
    ann = tifffile.imread(annotation_file)
    points = transformed_pnts_to_allen_helper_func(
        list(zyx_rois), ann, order="ZYX")

    # make dataframe
    lut_path =
    # "/jukebox/LightSheetTransfer/atlas/allen_atlas/allen_id_table.xlsx"
    df = count_structure_lister(lut_path, *points)
    df.to_excel(
        # "/jukebox/wang/willmore/lightsheet/20181213_fiber_optic_placement/processed/m278/allen_structures.xlsx")
