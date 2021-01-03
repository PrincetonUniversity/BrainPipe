# -*- coding: utf-8 -*-
"""
Example script to set up the parameters for the image processing pipeline
"""

######################### Import modules

import os, sys, pickle

from ClearMap.Utils.ParameterTools import joinParameter
from ClearMap.cluster.preprocessing import pth_update, listdirfull, makedir


def set_parameters_for_clearmap(testing=False, **kwargs):
    """TP: Wrapped this into a function such that parallel processes can get at variables
    
    testing should be set to false unless testing where it will save optimization files for cell detection.
    
    Editting of this function is how one changes cell detection parameters
    """
    
    #handle optional inputs, to change a parameter for your application, see run_clearmap_cluster.py. This is done before loading kwargs to ensure no overwrite during parameter sweep and/or cell detection testing
    rBP_size = kwargs["removeBackgroundParameter_size"] if "removeBackgroundParameter_size" in kwargs else (7,7)
    fEMP_hmax = kwargs["findExtendedMaximaParameter_hmax"] if "findExtendedMaximaParameter_hmax" in kwargs else None # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
    fEMP_size = kwargs["findExtendedMaximaParameter_size"] if "findExtendedMaximaParameter_size" in kwargs else 5 # size in pixels (x,y) for the structure element of the morphological opening
    fEMP_threshold = kwargs["findExtendedMaximaParameter_threshold"] if "findExtendedMaximaParameter_threshold" in kwargs else 0 # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
    fIP_method = kwargs["findIntensityParameter_method"] if "findIntensityParameter_method" in kwargs else "Max" ## (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
    fIP_size = kwargs["findIntensityParameter_size"] if "findIntensityParameter_size" in kwargs else (3,3,3) # (tuple)             size of the search box on which to perform the *method*
    dCSHP_threshold = kwargs["detectCellShapeParameter_threshold"] if "detectCellShapeParameter_threshold" in kwargs else 500 ## (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated
    
    
    ##input data from preprocessing:
    with open(os.path.join(kwargs["outputdirectory"], "param_dict.p"), "rb") as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()
    kwargs = pth_update(kwargs)
    
    vols=kwargs["volumes"]
    cfosvol = [xx for xx in vols if "cellch" in xx.ch_type][0]
    #lightsheet mods
    try:
        autovol = [xx for xx in vols if "regch" in xx.ch_type][0]
    except Exception as e:
        print(e)
        autovol = [xx for xx in vols if "cellch" in xx.ch_type][0]
    
    ######################### Data parameters
    
    #Data File and Reference channel File, usually as a sequence of files from the microscope
    #Use \d{4} for 4 digits in the sequence for instance. As an example, if you have cfos-Z0001.ome.tif :
    #os.path.join() is used to join the savedirectory path and the data paths:
    cFosFile = os.path.join(cfosvol.full_sizedatafld_vol, cfosvol.brainname + "_C" + cfosvol.channel + "_Z\d{4}.tif")
    AutofluoFile = os.path.join(autovol.full_sizedatafld_vol, autovol.brainname + "_C" + autovol.channel + "_Z\d{4}.tif")
    OriginalResolution = kwargs["xyz_scale"]
    
   
    #Specify the range for the cell detection. This doesn"t affect the resampling and registration operations
    cFosFileRange = {"x" : all, "y" : all, "z" : all};
    
    #Resolution of the Raw Data (in um / pixel)
    #OriginalResolution = (4.0625, 4.0625, 3);
    #OriginalResolution = (1.63, 1.63, 3); #multiplying everything by 2.5
    
    #Orientation: 1,2,3 means the same orientation as the reference and atlas files.
    #Flip axis with - sign (eg. (-1,2,3) flips x). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y)
    #FinalOrientation = (1,2,3);
    FinalOrientation = kwargs["FinalOrientation"] if "FinalOrientation" in kwargs else (3,2,1)
    
    #Resolution of the Atlas (in um/ pixel)
    AtlasResolution = kwargs["AtlasResolution"] if "AtlasResolution" in kwargs else (25, 25, 25)
    
    #Path to registration parameters and atlases
    AtlasFile      = kwargs["AtlasFile"] if "AtlasFile" in kwargs else "/jukebox/LightSheetTransfer/atlas/sagittal_atlas_20um_iso.tif"
    AnnotationFile = kwargs["annotationfile"] if "annotationfile" in kwargs else "/jukebox/LightSheetTransfer/atlas/annotation_sagittal_atlas_20um_iso.tif"
    
    ######################### Cell Detection Parameters using custom filters
    
    #Spot detection method: faster, but optimised for spherical objects.
    #You can also use "Ilastik" for more complex objects
    ImageProcessingMethod = "SpotDetection";
    
    #testing output:
    if testing:
        optdir = os.path.join(kwargs["outputdirectory"], "optimization"); makedir(optdir)
        sys.stdout.write("\nThis function is set for optimization of cell detection, optimization results in:\n   {}\n\n".format(optdir));sys.stdout.flush()
        bgdir = os.path.join(optdir, "background"); makedir(bgdir)
        bg = os.path.join(bgdir, "background\d{4}.ome.tif")
        exmaxdir = os.path.join(optdir, "extendmax"); makedir(exmaxdir)
        ex = os.path.join(exmaxdir, "extendmax\d{4}.ome.tif")
        celldir = os.path.join(optdir, "cell"); makedir(celldir)
        cell = os.path.join(celldir, "cell\d{4}.ome.tif")
        illum =  os.path.join(optdir, "illumination_correction")


    else:
        bg = None
        ex = None
        cell = None
        illum = None

            
    #For illumination correction (necessitates a specific calibration curve for your microscope)
    correctIlluminationParameter = {
        "flatfield"  : None,  # (True or None)  flat field intensities, if None do not correct image for illumination 
        "background" : None, # (None or array) background image as file name or array, if None background is assumed to be zero
        "scaling"    : None, # was "Mean" (str or None)        scale the corrected result by this factor, if "max"/"mean" scale to keep max/mean invariant
        "save"       : illum,       # (str or None)        save the corrected image to file
        "verbose"    : True    # (bool or int)        print / plot information about this step 
    }
    
    #Remove the background with morphological opening (optimised for spherical objects)
    removeBackgroundParameter = {
        "size"    : rBP_size,  # size in pixels (x,y) for the structure element of the morphological opening
        #"size"    : (7 * deltaresolution, 7 * deltaresolution),  # size in pixels (x,y) for the structure element of the morphological opening
        #"size"    : (5,5),  # size in pixels (x,y) for the structure element of the morphological opening # 5 seems very good for 1.3x objective, 5.5 is bad, 4.5 is really bad
        "save"    : bg,     # file name to save result of this operation
        "verbose" : True  # print / plot information about this step       
    }
    
    #Difference of Gaussians filter: to enhance the edges. Useful if the objects have a non smooth texture (eg: amyloid deposits)
    filterDoGParameter = {
        "size"    : None,        # (tuple or None)      size for the DoG filter in pixels (x,y,z) if None, do not correct for any background
        "sigma"   : None,        # (tuple or None)      std of outer Gaussian, if None automatically determined from size
        "sigma2"  : None,        # (tuple or None)      std of inner Gaussian, if None automatically determined from size
        "save"    : None,        # (str or None)        file name to save result of this operation if None dont save to file 
        "verbose" : True      # (bool or int)        print / plot information about this step
    }
    
    #Extended maxima: if the peak intensity in the object is surrounded by smaller peaks: avoids overcounting "granular" looking objects
    findExtendedMaximaParameter = {
        "hMax"      : fEMP_hmax,            # (float or None)     h parameter (for instance 20) for the initial h-Max transform, if None, do not perform a h-max transform
        "size"      : fEMP_size,             # (tuple)             size for the structure element for the local maxima filter
        "threshold" : fEMP_threshold,        # (float or None)     include only maxima larger than a threshold, if None keep all local maxima
        "save"      : ex,         # (str or None)       file name to save result of this operation if None dont save to file 
        "verbose"   : True       # (bool or int)       print / plot information about this step
    }
    
    #If no cell shape detection and the maximum intensity is not at the gravity center of the object, look for a peak intensity around the center of mass. 
    findIntensityParameter = {
        "method" : fIP_method,       # (str, func, None)   method to use to determine intensity (e.g. "Max" or "Mean") if None take intensities at the given pixels
        "size"   : fIP_size      # (tuple)             size of the search box on which to perform the *method*
    }
    
    #Object volume detection. The object is painted by a watershed, until reaching the intensity threshold, based on the background subtracted image
    detectCellShapeParameter = {
        "threshold" : dCSHP_threshold, #* deltaresolution,     # (float or None)      threshold to determine mask. Pixels below this are background if None no mask is generated. For 1.3x objective 500 is too large
        "save"      : cell,        # (str or None)        file name to save result of this operation if None dont save to file 
        "verbose"   : True      # (bool or int)        print / plot information about this step if None take intensities at the given pixels
    }
    
    #####
    
    ###########
    ###Nico"s parameters in paper w 4.0625:
    #background 7
    #threhold cell detection intensity of 700
    #cells within size o f20 to 500 voxels
    #density maps of voxelation of 15
    ###########
    
    ## Parameters for cell detection using spot detection algorithm 
    detectSpotsParameter = {
        "correctIlluminationParameter" : correctIlluminationParameter,
        "removeBackgroundParameter"    : removeBackgroundParameter,
        "filterDoGParameter"           : filterDoGParameter,
        "findExtendedMaximaParameter"  : findExtendedMaximaParameter,
        "findIntensityParameter"       : findIntensityParameter,
        "detectCellShapeParameter"     : detectCellShapeParameter
    }
    
    #set directories
    if not testing:
        savedirectory = os.path.join(kwargs["outputdirectory"], "clearmap_cluster_output")
        makedir(savedirectory)
        elastixdirectory = os.path.join(kwargs["parameterfolder"])
    else: 
        savedirectory = kwargs["outputdirectory"]
        elastixdirectory = os.path.join(kwargs["parameterfolder"])
    
    
    #################### Heat map generation
    
    ##Voxelization: file name for the output:
    VoxelizationFile = os.path.join(savedirectory, "points_voxelized.tif");
    
    # Parameter to calculate the density of the voxelization
    voxelizeParameter = {
        #Method to voxelize
        "method" : "Spherical", # Spherical,"Rectangular, Gaussian"
           
        # Define bounds of the volume to be voxelized in pixels
        "size" : (15,15,15),  
        #shouldn"t need to change as this is now at the level of the isotropic atlas
        #"size" : (15 * deltaresolution,15 * deltaresolution,15 * deltaresolution),
    
        # Voxelization weigths (e/g intensities)
        "weights" : None
    };
    
    ############################ Config parameters
    
    #Processes to use for Resampling (usually twice the number of physical processors)
    ResamplingParameter = {
        "processes": 16 
    };
      
    #Stack Processing Parameter for cell detection
    StackProcessingParameter = {
        #max number of parallel processes. Be careful of the memory footprint of each process!
        "processes" : 1, #automatically set to 1 if using processMethod
       
        #chunk sizes: number of planes processed at once
        "chunkSizeMax" : 20,
        "chunkSizeMin" : 5,
        "chunkOverlap" : 10,
    
        #optimize chunk size and number to number of processes to limit the number of cycles
        "chunkOptimization" : True,
        
        #increase chunk size for optimization (True, False or all = automatic)
        "chunkOptimizationSize" : all,
       
        "processMethod" : "cluster" #"sequential", "parallel"=local parallelization, "cluster" = parellization using cluster
       };
    
    ######################## Run Parameters, usually you don"t need to change those

    ### Resample Fluorescent and CFos images
    # Autofluorescent cFos resampling for aquisition correction
    
    ResolutionAffineCFosAutoFluo = kwargs["ResolutionAffineCFosAutoFluo"] if "ResolutionAffineCFosAutoFluo" in kwargs else (16, 16, 16)
    
    CorrectionResamplingParameterCfos = ResamplingParameter.copy();
    
    CorrectionResamplingParameterCfos["source"] = cFosFile;
    CorrectionResamplingParameterCfos["sink"]   = os.path.join(savedirectory, "cfos_resampled.tif");
        
    CorrectionResamplingParameterCfos["resolutionSource"] = OriginalResolution;
    CorrectionResamplingParameterCfos["resolutionSink"]   = ResolutionAffineCFosAutoFluo;
    
    CorrectionResamplingParameterCfos["orientation"] = FinalOrientation;
       
       
       
    #Files for Auto-fluorescence for acquisition movements correction
    CorrectionResamplingParameterAutoFluo = CorrectionResamplingParameterCfos.copy();
    CorrectionResamplingParameterAutoFluo["source"] = AutofluoFile;
    CorrectionResamplingParameterAutoFluo["sink"]   = os.path.join(savedirectory, "autofluo_for_cfos_resampled.tif");
       
    #Files for Auto-fluorescence (Atlas Registration)
    RegistrationResamplingParameter = CorrectionResamplingParameterAutoFluo.copy();
    RegistrationResamplingParameter["sink"]            =  os.path.join(savedirectory, "autofluo_resampled.tif");
    RegistrationResamplingParameter["resolutionSink"]  = AtlasResolution;
       
    
    ### Align cFos and Autofluo
    affinepth = [xx for xx in listdirfull(elastixdirectory) if "affine" in xx and "~" not in xx][0]
    bsplinepth = [xx for xx in listdirfull(elastixdirectory) if "bspline" in xx and "~" not in xx][0]
    
    
    CorrectionAlignmentParameter = {            
        #moving and reference images
        "movingImage" : os.path.join(savedirectory, "autofluo_for_cfos_resampled.tif"),
        "fixedImage"  : os.path.join(savedirectory, "cfos_resampled.tif"),
        
        #elastix parameter files for alignment
        "affineParameterFile"  : affinepth,
        "bSplineParameterFile" : bsplinepth,
        
        #directory of the alignment result
        "resultDirectory" :  os.path.join(savedirectory, "elastix_cfos_to_auto")
        }; 
      
    
    ### Align Autofluo and Atlas
    
    #directory of the alignment result
    RegistrationAlignmentParameter = CorrectionAlignmentParameter.copy();
    
    RegistrationAlignmentParameter["resultDirectory"] = os.path.join(savedirectory, "elastix_auto_to_atlas");
        
    #moving and reference images
    RegistrationAlignmentParameter["movingImage"]  = AtlasFile;
    RegistrationAlignmentParameter["fixedImage"]   = os.path.join(savedirectory, "autofluo_resampled.tif");
    
    #elastix parameter files for alignment
    RegistrationAlignmentParameter["affineParameterFile"] = affinepth
    RegistrationAlignmentParameter["bSplineParameterFile"] = bsplinepth
    
    # result files for cell coordinates (csv, vtk or ims)
    SpotDetectionParameter = {
        "source" : cFosFile,
        "sink"   : (os.path.join(savedirectory, "cells-allpoints.npy"),  os.path.join(savedirectory,  "intensities-allpoints.npy")),
        "detectSpotsParameter" : detectSpotsParameter
    };
    SpotDetectionParameter = joinParameter(SpotDetectionParameter, cFosFileRange)
    
    ImageProcessingParameter = joinParameter(StackProcessingParameter, SpotDetectionParameter);
    
    FilteredCellsFile = (os.path.join(savedirectory, "cells.npy"), os.path.join(savedirectory,  "intensities.npy"));
    
    TransformedCellsFile = os.path.join(savedirectory, "cells_transformed_to_Atlas.npy")
    
    ### Transform points from Original c-Fos position to autofluorescence
    
    ## Transform points from original to corrected
    # downscale points to referenece image size
    
    CorrectionResamplingPointsParameter = CorrectionResamplingParameterCfos.copy();
    CorrectionResamplingPointsParameter["pointSource"] = os.path.join(savedirectory, "cells.npy");
    CorrectionResamplingPointsParameter["dataSizeSource"] = cFosFile;
    CorrectionResamplingPointsParameter["pointSink"]  = None;
    
    CorrectionResamplingPointsInverseParameter = CorrectionResamplingPointsParameter.copy();
    CorrectionResamplingPointsInverseParameter["dataSizeSource"] = cFosFile;
    CorrectionResamplingPointsInverseParameter["pointSink"]  = None;
    
    ## Transform points from corrected to registered
    # downscale points to referenece image size
    RegistrationResamplingPointParameter = RegistrationResamplingParameter.copy();
    RegistrationResamplingPointParameter["dataSizeSource"] = cFosFile;
    RegistrationResamplingPointParameter["pointSink"]  = None;
    
    ##TP
    bigdct={}
    bigdct["detectSpotsParameter"] = detectSpotsParameter
    bigdct["voxelizeParameter"] = voxelizeParameter
    bigdct["ResamplingParameter"] = ResamplingParameter
    bigdct["StackProcessingParameter"] = StackProcessingParameter
    bigdct["CorrectionResamplingParameterCfos"] = CorrectionResamplingParameterCfos
    bigdct["CorrectionResamplingParameterAutoFluo"] = CorrectionResamplingParameterAutoFluo
    bigdct["RegistrationResamplingParameter"] = RegistrationResamplingParameter
    bigdct["CorrectionAlignmentParameter"] = CorrectionAlignmentParameter
    bigdct["RegistrationAlignmentParameter"]= RegistrationAlignmentParameter
    bigdct["SpotDetectionParameter"] = SpotDetectionParameter
    bigdct["ImageProcessingParameter"] = ImageProcessingParameter
    bigdct["FilteredCellsFile"] = FilteredCellsFile
    bigdct["TransformedCellsFile"] = TransformedCellsFile
    bigdct["CorrectionResamplingPointsParameter"] = CorrectionResamplingPointsParameter
    bigdct["CorrectionResamplingPointsInverseParameter"] = CorrectionResamplingPointsInverseParameter
    bigdct["RegistrationResamplingPointParameter"] = RegistrationResamplingPointParameter
    bigdct["AnnotationFile"] = AnnotationFile
    bigdct["VoxelizationFile"] = VoxelizationFile
    bigdct["ImageProcessingMethod"] = ImageProcessingMethod
    bigdct["AtlasFile"] = AtlasFile
    bigdct["OutputDirectory"] = kwargs["outputdirectory"]
    if testing: bigdct["OptimizationLocation"] = optdir
    #bigdct[] = 
    #bigdct[] = 
    
    
    return pth_update(bigdct)
