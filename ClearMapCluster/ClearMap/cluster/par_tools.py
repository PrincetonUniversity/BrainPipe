# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:37:28 2016

@author: tpisano
"""

import os, sys, shutil, numpy, tifffile, multiprocessing as mp
import pickle
from ClearMap.cluster.preprocessing import pth_update, makedir
import ClearMap.IO as io
from ClearMap.Alignment.Resampling import resampleData;
from ClearMap.Alignment.Elastix import alignData, transformPoints
from ClearMap.ImageProcessing.CellDetection import detectCells
from ClearMap.Alignment.Resampling import resamplePoints, resamplePointsInverse
from ClearMap.Analysis.Label import countPointsInRegions
from ClearMap.Analysis.Voxelization import voxelize
from ClearMap.Analysis.Statistics import thresholdPoints
from ClearMap.Analysis.Label import labelToName
from ClearMap.parameter_file import set_parameters_for_clearmap
from ClearMap.cluster.preprocessing import listdirfull
from ClearMap.ImageProcessing.StackProcessing import calculateSubStacks,\
    noProcessing, joinPoints
from scipy.ndimage.interpolation import zoom


def resample_folder(cores, inn, out, zoomfactor, compression=0):
    """Function to take as input folder of tiff images, resize and save

    Inputs
    ---------------------
    cores = number of parallel processses
    inn = input folder of tiffs
    out = output folder to save
    compression = (optional) compression factor
    """
    makedir(out)
    p = mp.Pool(cores)
    iterlst = []; [iterlst.append((out, inn, fl, zoomfactor, compression)) for fl in os.listdir(inn)]
    p.starmap(resample_folder_helper, iterlst)
    return

def resample_folder_helper(out, inn, fl, zoomfactor, compression):
    tifffile.imsave(os.path.join(out, fl), zoom(tifffile.imread(os.path.join(inn, fl)), zoomfactor), compress=compression)
    return

def resampling_operations(jobid, **params):
    """Assumes variables will be global from execfile step
    """
    #load
    dct=pth_update(set_parameters_for_clearmap(**params))

    #######################
    #resampling for the correction of stage movements during the acquisition between channels:
    if jobid == 0: resampleData(**dct["CorrectionResamplingParameterCfos"])
    if jobid == 1: resampleData(**dct["CorrectionResamplingParameterAutoFluo"])

    #Downsampling for alignment to the Atlas:
    if jobid == 2: resampleData(**dct["RegistrationResamplingParameter"])

    return

def alignment_operations(jobid, **params):
    """Assumes variables will be global from execfile step
    """
    #load
    dct=pth_update(set_parameters_for_clearmap(**params))

    #correction between channels:
    if jobid == 0: resultDirectory  = alignData(**dct["CorrectionAlignmentParameter"]);

    #alignment to the Atlas:
    if jobid == 1: resultDirectory  = alignData(**dct["RegistrationAlignmentParameter"]);

    return resultDirectory

def celldetection_operations(jobid, testing = False, **params):
    """Assumes variables will be global from execfile step

    testing = (optional) if "True" will save out different parameters. Only do this while optimizing
    """
    #load
    dct = pth_update(set_parameters_for_clearmap(testing=testing, **params))

    #set jobid
    dct["ImageProcessingParameter"]["jobid"]=jobid

    #detect cells
    result, substack = detectCells(**dct["ImageProcessingParameter"])
    if result == "ENDPROCESS": return "Jobid > # of jobs required, ending job"

    #save raw data for better comparision:
    if testing:
        rawfld = dct["OptimizationLocation"] + "/raw"; makedir(rawfld)
        substack["source"]
        for xx in range(substack["zCenterIndices"][0], substack["zCenterIndices"][1]):
            fl = substack["source"].replace(str("\\d{4}"), str(xx).zfill(4))
            shutil.copy2(fl, rawfld)

    return



def join_results_from_cluster(**params):

    #load
    dct=pth_update(set_parameters_for_clearmap(**params))

    #join_results
    out = join_results_from_cluster_helper(**dct["ImageProcessingParameter"])

    return out


def join_results_from_cluster_helper(source, x = all, y = all, z = all, sink = None,
                             chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                             function = noProcessing, join = joinPoints, verbose = False, **parameter):


    #calc num of substacks
    subStacks = calculateSubStacks(source, x = x, y = y, z = z,
                                   processes = 1, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,
                                   chunkOptimization = False, verbose = verbose);

    #load all cell detection job results
    if type(sink) == tuple: pckfld = os.path.join(sink[0][:sink[0].rfind("/")], "cell_detection"); makedir(pckfld)
    elif type(sink) == str: pckfld = os.path.join(sink[:sink.rfind("/")], "cell_detection"); makedir(pckfld)

    results = []; fls = listdirfull(pckfld); fls.sort()
    for fl in fls:
        if "~" not in fl:
            with open(fl, "rb") as pckl:
                results.append(pickle.load(pckl))
                pckl.close()
    print(results[0])
    #reformat
    results = [xx for yy in results for xx in yy]
    #join the results
    print ("Length of results: {}".format(len(results)))
    results = join(results, subStacks = subStacks, **parameter);

    #write / or return
    return io.writePoints(sink, results);



def output_analysis(threshold = (20, 900), row = (3,3), check_cell_detection = False, **params):
    """Wrapper for analysis:

    Inputs
    -------------------
    Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
    Row:
        row = (0,0) : peak intensity from the raw data
        row = (1,1) : peak intensity from the DoG filtered data
        row = (2,2) : peak intensity from the background subtracted data
        row = (3,3) : voxel size from the watershed

    Check Cell detection: (For the testing phase only, remove when running on the full size dataset)
    """
    dct=pth_update(set_parameters_for_clearmap(**params))

    points, intensities = io.readPoints(dct["ImageProcessingParameter"]["sink"]);

    #Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
    #row = (0,0) : peak intensity from the raw data
    #row = (1,1) : peak intensity from the DoG filtered data
    #row = (2,2) : peak intensity from the background subtracted data
    #row = (3,3) : voxel size from the watershed
    points, intensities = thresholdPoints(points, intensities, threshold = threshold, row = row);
    #points, intensities = thresholdPoints(points, intensities, threshold = (20, 900), row = (2,2));
    io.writePoints(dct["FilteredCellsFile"], (points, intensities));


    ## Check Cell detection (For the testing phase only, remove when running on the full size dataset)
    #######################
#    if check_cell_detection:
#        import ClearMap.Visualization.Plot as plt
#        pointSource= os.path.join(BaseDirectory, FilteredCellsFile[0]);
#        data = plt.overlayPoints(cFosFile, pointSource, pointColor = None, **cFosFileRange);
#        io.writeData(os.path.join(BaseDirectory, "cells_check.tif"), data);

    # Transform point coordinates
    #############################
    points = io.readPoints(dct["CorrectionResamplingPointsParameter"]["pointSource"]);
    points = resamplePoints(**dct["CorrectionResamplingPointsParameter"]);
    points = transformPoints(points, transformDirectory = dct["CorrectionAlignmentParameter"]["resultDirectory"], indices = False, resultDirectory = None);
    dct["CorrectionResamplingPointsInverseParameter"]["pointSource"] = points;
    points = resamplePointsInverse(**dct["CorrectionResamplingPointsInverseParameter"]);
    dct["RegistrationResamplingPointParameter"]["pointSource"] = points;
    points = resamplePoints(**dct["RegistrationResamplingPointParameter"]);
    points = transformPoints(points, transformDirectory = dct["RegistrationAlignmentParameter"]["resultDirectory"], indices = False, resultDirectory = None);
    io.writePoints(dct["TransformedCellsFile"], points);

    # Heat map generation
    #####################
    points = io.readPoints(dct["TransformedCellsFile"])
    intensities = io.readPoints(dct["FilteredCellsFile"][1])

    #Without weigths:
    vox = voxelize(points, dct["AtlasFile"], **dct["voxelizeParameter"]);
    if not isinstance(vox, str):
      io.writeData(os.path.join(dct["OutputDirectory"], "cells_heatmap.tif"), vox.astype("int32"));

    #With weigths from the intensity file (here raw intensity):
    dct["voxelizeParameter"]["weights"] = intensities[:,0].astype(float);
    vox = voxelize(points, dct["AtlasFile"], **dct["voxelizeParameter"]);
    if not isinstance(vox, str):
      io.writeData(os.path.join(dct["OutputDirectory"], "cells_heatmap_weighted.tif"), vox.astype("int32"));

    #Table generation:
    ##################
    #With integrated weigths from the intensity file (here raw intensity):
    try:
        ids, counts = countPointsInRegions(points, labeledImage = dct["AnnotationFile"], intensities = intensities, intensityRow = 0);
        table = numpy.zeros(ids.shape, dtype=[("id","int64"),("counts","f8"),("name", "a256")])
        table["id"] = ids;
        table["counts"] = counts;
        table["name"] = labelToName(ids);
        io.writeTable(os.path.join(dct["OutputDirectory"], "Annotated_counts_intensities.csv"), table);

        #Without weigths (pure cell number):
        ids, counts = countPointsInRegions(points, labeledImage = dct["AnnotationFile"], intensities = None);
        table = numpy.zeros(ids.shape, dtype=[("id","int64"),("counts","f8"),("name", "a256")])
        table["id"] = ids;
        table["counts"] = counts;
        table["name"] = labelToName(ids);
        io.writeTable(os.path.join(dct["OutputDirectory"], "Annotated_counts.csv"), table);
    except:
        print("Table not generated.\n")

    print ("Analysis Completed")

    return

def group_output_analysis(lst, threshold = (20, 900), row = (3,3), check_cell_detection = False):
    """Perform analysis on a group of files

    Inputs
    -------------------
    Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
    Row:
        row = (0,0) : peak intensity from the raw data
        row = (1,1) : peak intensity from the DoG filtered data
        row = (2,2) : peak intensity from the background subtracted data
        row = (3,3) : voxel size from the watershed

    Check Cell detection: (For the testing phase only, remove when running on the full size dataset)


    e.g.
    lst = ["/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_b",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_c",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_d",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_e"]

       group_output_analysis(lst, threshold = (20, 900), row = (2,2), check_cell_detection = False)

    """

    for xx in lst:
        #
        sys.stdout.write("\n\nStarting analysis on {}...".format(xx)); sys.stdout.flush()
        #load
        params = load_kwargs(xx)

        #run
        output_analysis(threshold = threshold, row = row, check_cell_detection = check_cell_detection, **params)

        sys.stdout.write("\n\nCompleted analysis on {}...".format(xx)); sys.stdout.flush()


    return

def group_output_analysis_par(lst, threshold = (20, 900), row = (3,3), check_cell_detection = False, cores=10):
    """Perform analysis on a group of files - parallelized

    Inputs
    -------------------
    Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
    Row:
        row = (0,0) : peak intensity from the raw data
        row = (1,1) : peak intensity from the DoG filtered data
        row = (2,2) : peak intensity from the background subtracted data
        row = (3,3) : voxel size from the watershed

    Check Cell detection: (For the testing phase only, remove when running on the full size dataset)


    e.g.
    lst = ["/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_b",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_c",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_d",
       "/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_e"]

       group_output_analysis(lst, threshold = (20, 900), row = (2,2), check_cell_detection = False)

    """
    p=mp.Pool(cores)
    iterlst = [(xx, threshold, row, check_cell_detection) for xx in lst]
    p.map(group_output_analysis_par_helper, iterlst)
    p.terminate()

    return

def group_output_analysis_par_helper(input_list):
    """helper for above
    """
    src, threshold, row, check_cell_detection = input_list

    #
    sys.stdout.write("\n\nStarting analysis on {}...".format(src)); sys.stdout.flush()
    #load
    params = load_kwargs(src)

    #run
    output_analysis(threshold = threshold, row = row, check_cell_detection = check_cell_detection, **params)

    sys.stdout.write("\n\nCompleted analysis on {}...".format(src)); sys.stdout.flush()


    return



def load_kwargs(outdr):
    """simple function to load kwargs given an "outdr"
    """
    kwargs = {}; kwargs = dict([("outputdirectory",outdr)])
    with open(pth_update(os.path.join(kwargs["outputdirectory"], "param_dict.p")), "rb") as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return pth_update(kwargs)
