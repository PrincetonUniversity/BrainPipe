#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 2020
@author: emilyjanedennis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.conv_net.utils.io import pairwise_distance_metrics, read_roi_zip

pth = "/path/to/annotated_volumes"
dst = "/path/to/resultsfolder"

flds = [os.path.join(pth, "sweep/"+xx) for xx in os.listdir(os.path.join(pth, "sweep"))
        if "max" in xx]  # only using the ones from the new sweep

roi_pths = [os.path.join(pth, xx) for xx in os.listdir(pth) if "RoiSet.zip" in xx]
roi_pths.sort()  # the sorts here are important, for order of volumes

# these will be zyx
# note that importing it this way, the z dimension does not start from 0, but neither does clearmaps, so probably ok???
annotated_cells = np.array([np.array([[int(yy) for yy in xx[0].replace(".roi", "").split("-")] for xx in
                                      read_roi_zip(roi_pth, include_roi_name=True)]) for roi_pth in roi_pths])


# voxel pair cutoff
cutoffs = [5, 7, 10]


for cutoff in cutoffs:
    # init dataframe
    print("cutoff: %s \n\n" % cutoff)
    df = pd.DataFrame()
    df["parameters"] = [os.path.basename(xx) for xx in flds]

    df["tp_f37104_demonstrator"] = np.zeros(len(df))
    df["tp_m37079_mouse2"] = np.zeros(len(df))
    df["tp_f37077_observer"] = np.zeros(len(df))

    df["tp_f37104_demonstrator"] = np.zeros(len(df))
    df["fp_m37079_mouse2"] = np.zeros(len(df))
    df["fp_f37077_observer"] = np.zeros(len(df))

    df["fn_f37104_demonstrator"] = np.zeros(len(df))
    df["fn_m37079_mouse2"] = np.zeros(len(df))
    df["fn_f37077_observer"] = np.zeros(len(df))

    df["f1_f37104_demonstrator"] = np.zeros(len(df))
    df["f1_m37079_mouse2"] = np.zeros(len(df))
    df["f1_f37077_observer"] = np.zeros(len(df))

    for n, fld in enumerate(flds):
        if n % 100 == 0:
            print(n)
        arr_pths = [os.path.join(fld, xx) for xx in os.listdir(
            fld) if not os.path.isdir(os.path.join(fld, xx))]
        arr_pths.sort()  # the sorts here are important, for order of volumes
        detected_cells = np.asarray([np.load(arr_pth) for arr_pth in arr_pths])
        measures = [pairwise_distance_metrics(annotated_cells[i], detected_cells[i], cutoff=cutoff, verbose=False) for i
                    in range(len(arr_pths))]

        for m, measure in enumerate(measures):
            paired, tp, fp, fn = measure
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)  # calculating precision and recall
                f1 = 2*((precision*recall)/(precision+recall))  # calculating f1 score
            except Exception as e:
                print(e)
                f1 = np.nan  # if tp, fn, etc. are 0
            if m == 0:  # these will always be in this order, hence saving this way, easier than doin this auto with the long filenames
                df.loc[df.parameters == os.path.basename(fld), "f1_f37104_demonstrator"] = f1
                df.loc[df.parameters == os.path.basename(fld), "tp_f37104_demonstrator"] = tp
                df.loc[df.parameters == os.path.basename(fld), "fp_f37104_demonstrator"] = fp
                df.loc[df.parameters == os.path.basename(fld), "fn_f37104_demonstrator"] = fn
            if m == 1:
                df.loc[df.parameters == os.path.basename(fld), "f1_m37079_mouse2"] = f1
                df.loc[df.parameters == os.path.basename(fld), "tp_m37079_mouse2"] = tp
                df.loc[df.parameters == os.path.basename(fld), "fp_m37079_mouse2"] = fp
                df.loc[df.parameters == os.path.basename(fld), "fn_m37079_mouse2"] = fn
            if m == 2:
                df.loc[df.parameters == os.path.basename(fld), "f1_f37077_observer"] = f1
                df.loc[df.parameters == os.path.basename(fld), "tp_f37077_observer"] = tp
                df.loc[df.parameters == os.path.basename(fld), "fp_f37077_observer"] = fp
                df.loc[df.parameters == os.path.basename(fld), "fn_f37077_observer"] = fn

df.to_csv(os.path.join(dst, "clearmap_voxelcutoff_%02d_v2.csv" % cutoff))
