import os
from tools.utils.directorydeterminer import directorydeterminer

systemdirectory=directorydeterminer() # root path of your filesystem
###set paths to data
###inputdictionary stucture: key=pathtodata, value=list["xx", "##"] where xx=regch, injch, or cellch and ##=two digit channel number
#"regch" = channel to be used for registration, assumption is all other channels are signal
#"cellch" = channel(s) to apply cell detection
#"injch" = channels(s) to quantify injection site
#e.g.: inputdictionary={path_1: [["regch", "00"]], path_2: [["cellch", "00"], ["injch", "01"]]} ###create this dictionary variable BEFORE params
inputdictionary={
os.path.join(systemdirectory, 
    "/jukebox/LightSheetData/lavision_testdata/4x_example/190430_m57206_obs_cfos_20190319_4x_647_008na_1hfds_z2um_200msec_10povlp_10-59-43"): 
    [["regch","00"]]
}

####Required inputs
params={
"systemdirectory":  systemdirectory, #don"t need to touch
"inputdictionary": inputdictionary, #don"t need to touch
"outputdirectory": os.path.join(systemdirectory, "wang/ahoag/test_stitching2_brainpipe_forpub"),
"xyz_scale": (1.63,1.63,2), #micron/pixel x,y,z or raw data
"tiling_overlap": 0.1, #percent overlap taken during tiling, disregard if images not tiled
"stitchingmethod": "terastitcher", # "terastitcher" if stitching needed, otherwise use "blending" see below for details
"AtlasFile": os.path.join(systemdirectory, "LightSheetTransfer/atlas/sagittal_atlas_20um_iso.tif"),
"annotationfile": os.path.join(systemdirectory, "LightSheetTransfer/atlas/annotation_sagittal_atlas_20um_iso_16bit.tif"), ###path to annotation file for structures
"blendtype": "sigmoidal", #False/None, "linear", or "sigmoidal" blending between tiles, usually sigmoidal; False or None for images where blending would be detrimental
"intensitycorrection": True, #True = calculate mean intensity of overlap between tiles shift higher of two towards lower - useful for images where relative intensity is not important (i.e. tracing=True, cFOS=False)
"resizefactor": 5, ##in x and y #normally set to 5 for 4x objective, 3 for 1.3x obj
"rawdata": True, # set to true if raw data is taken from scope and images need to be flattened; functionality for rawdata =False has not been tested**
"finalorientation":  ("2","1","0"), #Used to account for different orientation between brain and atlas. Assumes XYZ ("0","1","2) orientation. Pass strings NOT ints. "-0" = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation import fix_orientation; ("2","1","0") for horizontal to sagittal, Order of operations is reversing of axes BEFORE swapping axes.
"slurmjobfactor": 50, #number of array iterations per arrayjob since max job array on SPOCK is 1000
"transfertype": "copy",
"parameterfolder": "parameterfolder" 
}

#####################################################################################################################################################
##################################################stitchingmethod####################################################################################
#####################################################################################################################################################
# "terastitcher": computationally determine overlap. See .py file and http://abria.github.io/TeraStitcher/ for details. NOTE THIS REQUIRES COMPILED SOFTWARE.
    #if testing terastitcher I strongly suggest adding to the parameter file
    #transfertype="copy", despite doubling data size this protects original data while testing
#"blending: using percent overlap to determine pixel overlap. Then merges using blendtype, intensitycorrection, blendfactor. This is not a smart algorithm
    
#####################################################################################################################################################
##################################################optional arguments for params######################################################################
#####################################################################################################################################################
# "regexpression":  r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", ###lavision preprocessed data
# "regexpression":  r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", lavision NONTILING + nonraw**
# "regexpression":  r"(.*)(.*C+)(.*)(.*Z+)(?P<z>[0-9]{1,4})(.*r+)(?P<ch>[0-9]{1,4})(.ome.tif)",
# "parameterfolder" : os.path.join(systemdirectory, "wang/pisano/Python/lightsheet/parameterfolder"), ##  * folder consisting of elastix parameter files with prefixes "Order<#>_" to specify application order
# "atlas_scale": (25, 25, 25), #micron/pixel, ABA is likely (25,25,25)
# "swapaxes" :  (0,2), #Used to account for different orientation between brain and atlas. 0=z, 1=y, 2=x. i.e. to go from horizontal scan to sagittal (0,2).
# "maskatlas": {"x": all, "y": "125:202", "z": "75:125"}; dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be zeroed out. Occurs AFTER orientation change.
# "cropatlas": {"x": all, "y": "125:202", "z": "75:125"}; dictionary consisting of x,y,z ranges of atlas to keep, the rest of the atlas will be REMOVED rather than zeroed out. THIS FUNCTION DOES NOT YET AFFECT THE ANNOTATION FILE
# "blendfactor" : 4, #only for sigmoidal blending, controls the level of sigmoidal; parameter that is passed to np"s linspace; defaults to 4. Higher numbers = steeper blending; lower number = more gradual blending
# "bitdepth": specify the fullsizedatafolder bitdepth output
# "secondary_registration" True (default) - register other channel(s) to registration channel (regch) then apply transform determined from regch->atlas
#                          useful if imaging conditions were different between channel and regch, i.e. added horizontal foci, sheet na...etc
#                          False - apply transform determined from regch->atlas to other channels. Use if channel of interest has very different pixel distribution relative regch (i.e. dense labeling)
