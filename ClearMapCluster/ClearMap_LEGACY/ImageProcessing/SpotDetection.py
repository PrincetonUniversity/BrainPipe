# -*- coding: utf-8 -*-
"""
Functions to detect spots in images

The main routine :func:`detectCells` uses a difference of gaussian filter (see 
:mod:`~ClearMap.ImageProcessing.Filter`) followed by a peak detection step.

Example:

    >>> import os
    >>> import ClearMap.IO as io  
    >>> import ClearMap.Settings as settings
    >>> import ClearMap.ImageProcessing.SpotDetection as sd
    >>> fn = os.path.join(settings.ClearMapPath, "Test/Data/Synthetic/test_iDISCO_\d{3}.tif");
    >>> img = io.readData(fn);
    >>> img = img.astype("int16"); # converting data to smaller integer types can be more memory efficient!
    >>> res = sd.detectSpots(img, dogSize = (5,5,5), flatfield = None, threshold = 5, cellShapeThreshold = 1);
    >>> print "Found %d cells !" % res[0].shape[0]
    Illumination: flatfield          : None
    Illumination: illuminationScaling: True
    Illumination: background         : None
    Background: backgroundSize: (15, 15)
    Background: elapsed time: 0:00:00
    DoG: dogSize: (5, 5, 5)
    DoG: elapsed time: 0:00:00
    Extended Max: threshold   : 5
    Extended Max: localMaxSize: 5
    Extended Max: hMax        : None
    Extended Max: elapsed time: 0:00:00
    Cell Centers: elapsed time: 0:00:00
    Cell Shape: cellShapeThreshold: 1
    Cell Shape:: elapsed time: 0:00:00
    Cell Size:: elapsed time: 0:00:00
    Cell Intensity: cellIntensityMethod: Max
    Cell Intensity:: elapsed time: 0:00:00
    Cell Intensity: cellIntensityMethod: Max
    Cell Intensity:: elapsed time: 0:00:00
    Cell Intensity: cellIntensityMethod: Max
    Cell Intensity:: elapsed time: 0:00:00
    Found 38 cells !
    
After execution this example inspect the result of the cell detection in 
the folder "Test/Data/CellShape/cellshape\_\\d{3}.tif".
"""
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.

import sys, imp
import numpy


from ClearMap.ImageProcessing.IlluminationCorrection import correctIllumination
from ClearMap.ImageProcessing.BackgroundRemoval import removeBackground
from ClearMap.ImageProcessing.Filter.DoGFilter import filterDoG
from ClearMap.ImageProcessing.MaximaDetection import findExtendedMaxima, findPixelCoordinates, findIntensity, findCenterOfMaxima
from ClearMap.ImageProcessing.CellSizeDetection import detectCellShape, findCellSize, findCellIntensity

from ClearMap.Utils.Timer import Timer
from ClearMap.Utils.ParameterTools import getParameter


##############################################################################
# Spot detection
##############################################################################

def detectSpots(img, detectSpotsParameter = None, correctIlluminationParameter = None, removeBackgroundParameter = None,
                filterDoGParameter = None, findExtendedMaximaParameter = None, detectCellShapeParameter = None,
                verbose = False, out = sys.stdout, **parameter):
    """Detect Cells in 3d grayscale image using DoG filtering and maxima detection
    
    Effectively this function performs the following steps:
        * illumination correction via :func:`~ClearMap.ImageProcessing.IlluminationCorrection.correctIllumination`
        * background removal via :func:`~ClearMap.ImageProcessing.BackgroundRemoval.removeBackground`
        * difference of Gaussians (DoG) filter via :func:`~ClearMap.ImageProcessing.Filter.filterDoG`
        * maxima detection via :func:`~ClearMap.ImageProcessing.MaximaDetection.findExtendedMaxima`
        * cell shape detection via :func:`~ClearMap.ImageProcessing.CellSizeDetection.detectCellShape`
        * cell intensity and size measurements via: :func:`~ClearMap.ImageProcessing.CellSizeDetection.findCellIntensity`,
          :func:`~ClearMap.ImageProcessing.CellSizeDetection.findCellSize`. 
    
    Note: 
        Processing steps are done in place to save memory.
        
    Arguments:
        img (array): image data
        detectSpotParameter: image processing parameter as described in the individual sub-routines
        verbose (bool): print progress information
        out (object): object to print progress information to
        
    Returns:
        tuple: tuple of arrays (cell coordinates, raw intensity, fully filtered intensty, illumination and background corrected intensity [, cell size])
    """

    timer = Timer();
    
    # correct illumination
    correctIlluminationParameter = getParameter(detectSpotsParameter, "correctIlluminationParameter", correctIlluminationParameter);
    img1 = img.copy();
    img1 = correctIllumination(img1, correctIlluminationParameter = correctIlluminationParameter, 
                               verbose = verbose, out = out, **parameter)   

    # background subtraction in each slice
    #img2 = img.copy();
    removeBackgroundParameter = getParameter(detectSpotsParameter, "removeBackgroundParameter", removeBackgroundParameter);
    img2 = removeBackground(img1, removeBackgroundParameter = removeBackgroundParameter, 
                            verbose = verbose, out = out, **parameter)   

    #DoG filter
    filterDoGParameter = getParameter(detectSpotsParameter, "filterDoGParameter", filterDoGParameter);
    dogSize = getParameter(filterDoGParameter, "size", None);
    #img3 = img2.copy();    
    img3 = filterDoG(img2, filterDoGParameter = filterDoGParameter, verbose = verbose, out = out, **parameter);
    
    # extended maxima
    findExtendedMaximaParameter = getParameter(detectSpotsParameter, "findExtendedMaximaParameter", findExtendedMaximaParameter);
    hMax = getParameter(findExtendedMaximaParameter, "hMax", None);
    imgmax = findExtendedMaxima(img3, findExtendedMaximaParameter = findExtendedMaximaParameter,
                                verbose = verbose, out = out, **parameter);
    
    #center of maxima
    if not hMax is None:
        centers = findCenterOfMaxima(img, imgmax, verbose = verbose, out = out, **parameter);
    else:
        centers = findPixelCoordinates(imgmax, verbose = verbose, out = out, **parameter);
    
    #cell size detection
    detectCellShapeParameter = getParameter(detectSpotsParameter, "detectCellShapeParameter", detectCellShapeParameter);
    cellShapeThreshold = getParameter(detectCellShapeParameter, "threshold", None);
    if not cellShapeThreshold is None:
        
        # cell shape via watershed
        imgshape = detectCellShape(img2, centers, detectCellShapeParameter = detectCellShapeParameter, verbose = verbose, out = out, **parameter);
        
        #size of cells        
        csize = findCellSize(imgshape, maxLabel = centers.shape[0], out = out, **parameter);
        
        #intensity of cells
        cintensity = findCellIntensity(img, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);

        #intensity of cells in background image
        cintensity2 = findCellIntensity(img2, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);
    
        #intensity of cells in dog filtered image
        if dogSize is None:
            cintensity3 = cintensity2;
        else:
            cintensity3 = findCellIntensity(img3, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter);
        
        if verbose:
            out.write(timer.elapsedTime(head = "Spot Detection") + "\n");
        
        #remove cell;s of size 0
        idz = csize > 0;
                       
        return ( centers[idz], numpy.vstack((cintensity[idz], cintensity3[idz], cintensity2[idz], csize[idz])).transpose());        
        
    
    else:
        #intensity of cells
        cintensity = findIntensity(img, centers, verbose = verbose, out = out, **parameter);

        #intensity of cells in background image
        cintensity2 = findIntensity(img2, centers, verbose = verbose, out = out, **parameter);
    
        #intensity of cells in dog filtered image
        if dogSize is None:
            cintensity3 = cintensity2;
        else:
            cintensity3 = findIntensity(img3, centers, verbose = verbose, out = out, **parameter);

        if verbose:
            out.write(timer.elapsedTime(head = "Spot Detection") + "\n");
    
        return ( centers, numpy.vstack((cintensity, cintensity3, cintensity2)).transpose());
        
#%%
if __name__ == "__main__": 
    
    import os
    import ClearMap.ImageProcessing.SpotDetection as self
    imp.reload(self)
    import ClearMap.IO as io  
    import numpy as np
    import tifffile
    
    #run it on cfos volumes
    pth = "/jukebox/wang/zahra/kelly_cell_detection_analysis/comparison_to_clearmap/annotated_volumes"
    vols = [os.path.join(pth, xx) for xx in os.listdir(pth) if "tif" in xx]
    
    #sweep
    max_thresholds = np.arange(2,10,2)
    DoGs = np.arange(2,16,2)
    backgrounds = np.arange(3, 13, 2)
    sizes = np.arange(2, 22, 2)
    
    for max_threshold in max_thresholds:
        for DoG in DoGs:
            for background in backgrounds:
                for size in sizes:
                    dst = os.path.join(pth, "sweep/max_threshold%02d_DoG%02d_size%02d_background%02d" % (max_threshold,
                                                             DoG, size, background))
                    
                    if not os.path.exists(dst): 
                        os.mkdir(dst)
                        #save tifs elsewhere
                        vis = os.path.join(dst, "stacks")
                        if not os.path.exists(vis): os.mkdir(vis)
                        
                        print("\n\n{}".format(size))
                        for fn in vols:
                            img = io.readData(fn) #read as x,y,z
                            img = img.astype("uint16")
                            
                            c = detectSpots(img, detectSpotsParameter = None, correctIlluminationParameter = None, 
                                    removeBackgroundParameter = {"size": (background, background)},
                                    filterDoGParameter = {"size": (DoG, DoG, 4)}, findExtendedMaximaParameter = {"h-max": None, 
                                                          "size": size, "threshold": max_threshold},
                                    findIntensityParameter = {"size": (3,3,3), "method": "Max"},
                                    detectCellShapeParameter = {"threshold": None}, verbose = False)
                            
                            print("done, found %d cells !" % c[0].shape[0])
                            
                            cells_reshape = np.array([[z,y,x] for x,y,z in c[0]]) #now z,y,x
                            
                            np.save(os.path.join(dst, os.path.basename(fn)[:-8]+"_clearmap.npy"), cells_reshape)
                            
                            cell_map = np.zeros_like(tifffile.imread(fn))
                            
                            for z,y,x in cells_reshape:
                                cell_map[z,y,x] = 45000
                            
                            tifffile.imsave(os.path.join(vis, os.path.basename(fn)[:-8]+"_clearmap.tif"), cell_map.astype("float32"))
                

#%%  
    from skimage.morphology import ball
    import cv2

    #run on 4x volume
    dst = "/jukebox/wang/Jess/lightsheet_output/201904_ymaze_cfos/clearmap_accuracy_quantification/4x_cell_count_merged_maps"
    vol = "/home/wanglab/Desktop/an19_ymazefos_020719_4x_striatum.tif"
    
    #reading in the native z,y,x orientation
    vol = tifffile.imread(vol)
    
    #dtype
    img = vol.astype("uint16")
    
    for i in np.arange(0, vol.shape[0], 20):
        c = detectSpots(img[i:i+20], detectSpotsParameter = None, correctIlluminationParameter = None, 
                removeBackgroundParameter = {"size": (5,5)},
                filterDoGParameter = None, findExtendedMaximaParameter = {"h-max": None, "size": 20, "threshold": 0},
                findIntensityParameter = {"size": (5,5,5), "method": "Max"},
                detectCellShapeParameter = {"threshold": 150}, verbose = False)
        
        print("done, found %d cells !" % c[0].shape[0])
    
        cell_map = np.zeros_like(img[i:i+20])
        
        for z,y,x in c[0]:
            cell_map[z,y,x] = 45000
        
        #dilate
        r = 3
        selem = ball(r)[int(r/2)]
        cell_map = cell_map.astype("uint8")
        cell_map = np.asarray([cv2.dilate(cell_map[j], selem, iterations = 1) for j in range(cell_map.shape[0])])
            
        #save out slices
        merged = np.stack([img[i+1:i+20], cell_map[1:].astype("uint16"), np.zeros_like(cell_map[1:])], -1)
        tifffile.imsave(os.path.join(dst, "an19_ymazefos_020719_4x_striatum_test_{}_clearmap.tif".format(i)), merged)
    
    
