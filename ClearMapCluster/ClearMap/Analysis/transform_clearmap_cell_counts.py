#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:16:03 2019

@author: wanglab
"""

import os, numpy as np, pandas as pd, xlrd, tifffile, SimpleITK as sitk

def labelPoints(points, labeledImage, level = None, collapse = None):
   """ borrowed/modified/cleaned up from Clearmap """

   x = points[:,0];
   y = points[:,1];
   z = points[:,2];

   nPoint = x.size;

   pointLabels = np.zeros(nPoint, 'int32');

   labelImage = sitk.GetArrayFromImage(sitk.ReadImage(labeledImage))
   dsize = labelImage.shape;

   for i in range(nPoint):

       if x[i] >= 0 and x[i] < dsize[0] and y[i] >= 0 and y[i] < dsize[1] and z[i] >= 0 and z[i] < dsize[2]:
            pointLabels[i] = labelImage[int(x[i]), int(y[i]), int(z[i])];

   return pointLabels

def countPointsInRegions(points, labeledImage, intensities = None, intensityRow = 0, level= None, sort = True,
                         returnIds = True, returnCounts = False, collapse = None):

    """ borrowed/modified/cleaned up from Clearmap """
    pointLabels = labelPoints(points, labeledImage, level = level, collapse = collapse);

    if intensities is None:
        ll, cc = np.unique(pointLabels, return_counts = True);
        cci = None;
    else:
        if intensities.ndim > 1:
            intensities = intensities[:,intensityRow];

        ll, ii, cc = np.unique(pointLabels, return_counts = True, return_inverse = True);
        cci = np.zeros(ll.shape);
        for i in range(ii.shape[0]):
             cci[ii[i]] += intensities[i];

    #cc = numpy.vstack((ll,cc)).T;
    if sort:
        ii = np.argsort(ll);
        cc = cc[ii];
        ll = ll[ii];
        if not cci is None:
            cci = cci[ii];

    if returnIds:
        if cci is None:
            return ll, cc
        else:
            if returnCounts:
                return ll, cc, cci;
            else:
                return ll, cci
    else:
        if cci is None:
            return cc;
        else:
            if returnCounts:
                return cc, cci;
            else:
                return cci;

def make_table_of_transformed_cells(src, ann, ann_lut):
    """
    incorporates new atlas and look up table for anatomical analysis of cfos data done using:
        https://github.com/PrincetonUniversity/ClearMap
    NOTE: this assumes the clearmap transform has run correctly, and uses transformed cells
    """

    print(src)
    #first, check if cells where transformed properly
    #TRANSFORMED cell dataframe
    try:
        points = np.load(os.path.join(src, "clearmap_cluster_output/cells_transformed_to_Atlas.npy"))
        raw = np.load(os.path.join(src, "clearmap_cluster_output/cells.npy"))

        print(points.shape, raw.shape)

        if points.shape == raw.shape:
            intensities = np.load(os.path.join(src, "clearmap_cluster_output/intensities.npy"))

            #open LUT excel sheet
            wb = xlrd.open_workbook(ann_lut)
            lut = wb.sheet_by_index(0)

            #Table generation:
            ##################
            #With integrated weigths from the intensity file (here raw intensity):
            ids, intensities = countPointsInRegions(points, labeledImage = ann, intensities = intensities, intensityRow = 0)
            #keep them together to modify together later
            ids_intensities = list(zip(ids.astype("float64"), intensities))

            #mapping
            #NOTE: REMOVES "basic cell groups and region since LUT doesn"t have a value for that. FIX!??!
            id2name = dict((row[3].value, row[1]) for row in (lut.row(r) for r in range(lut.nrows))) #messy way to do things but works
            id2parent = dict((row[3].value, row[6]) for row in (lut.row(r) for r in range(lut.nrows)))
            id2acronym = dict((row[3].value, row[2]) for row in (lut.row(r) for r in range(lut.nrows)))
            id2parentacr = dict((row[3].value, row[7]) for row in (lut.row(r) for r in range(lut.nrows)))
            id2voxcount = dict((row[3].value, row[8]) for row in (lut.row(r) for r in range(lut.nrows)))

            lut_ids = [row[3].value for row in (lut.row(r) for r in range(lut.nrows))]

            table = {}
            table["id"] = [i_d[0] for i_d in ids_intensities if i_d[0] in lut_ids]
            table["intensity"] = [i_d[1] for i_d in ids_intensities if i_d[0] in lut_ids]

            #dropping structures that do not map to LUT
            table["name"] = [id2name[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["acronym"] = [id2acronym[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["parent_name"] = [id2parent[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["parent_acronym"] = [id2parentacr[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["voxels_in_structure"] = [id2voxcount[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]

            pd.DataFrame.from_dict(table, orient = "columns").to_csv(os.path.join(src, "Annotated_counts_intensities.csv"))

            #Without weigths (pure cell number):
            ids, counts = countPointsInRegions(points, labeledImage = ann, intensities = None)
            ids_counts = list(zip(ids.astype("float64"), counts))

            table = {}
            table["id"] = [i_d[0] for i_d in ids_counts if i_d[0] in lut_ids]
            table["counts"] = [i_d[1] for i_d in ids_counts if i_d[0] in lut_ids]

            #dropping structures that do not map to LUT
            table["name"] = [id2name[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["acronym"] = [id2acronym[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["parent_name"] = [id2parent[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["parent_acronym"] = [id2parentacr[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]
            table["voxels_in_structure"] = [id2voxcount[i_d].value for i_d in ids[1:] if i_d in id2name.keys()]

            pd.DataFrame.from_dict(table, orient = "columns").to_csv(os.path.join(src, "Annotated_counts.csv"))

            print ("\n Analysis Completed\n")
        else:
            print ("\n Cells not transformed properly, check transformix (aka step 6) and try again\n")
    except:
        print("\n Path for transformed cells doesn't exist\n")


if __name__ == "__main__":
    #goal is to transform cooridnates, voxelize based on number of cells and overlay with reigstered cell signal channel...
    #inputs

    #LUT
    ann = "/jukebox/LightSheetTransfer/atlas/allen_atlas/annotation_2017_25um_sagittal_forDVscans.nrrd"
    ann_lut = "/jukebox/LightSheetTransfer/atlas/allen_atlas/allen_id_table_w_voxel_counts.xlsx"

    pth = "/jukebox/LightSheetData/falkner-mouse/scooter/clearmap_processed"

    #for src in os.listdir(pth):
    src = "fmnp5"
    make_table_of_transformed_cells(os.path.join(pth, src), ann, ann_lut)
