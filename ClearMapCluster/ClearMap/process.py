# -*- coding: utf-8 -*-
"""
Template to run the processing pipeline
"""

##TP
import sys, tifffile, os
from ClearMap.cluster import par_tools, preprocessing
systemdirectory=preprocessing.directorydeterminer()
cwd = os.getcwd()
sys.path.append('/home/wanglab/wang/pisano/Python/lightsheet')
from tools.utils import resample_folder
import time
from ClearMap.parameter_file import set_parameters_for_clearmap

inn_cfos = '/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_A/full_sizedatafld/20150828_cfos_A_555_3hfds_100msec_z3um_ch00'
out_cfos = '/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_A/full_sizedatafld/downsized_555'

resample_folder(12, inn_cfos, out_cfos, .4, compression=1)

inn_auto = '/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_A/full_sizedatafld/20150828_cfos_A_488_1hfds_200msec_z3um_ch00'
out_auto = '/home/wanglab/wang/pisano/tracing_output/cfos/20150828_cfos_A/full_sizedatafld/downsized_488'

resample_folder(12, inn_auto, out_auto, .4, compression=1)

#%%

#load the parameters:
execfile(os.path.join(cwd, 'ClearMap', 'parameter_file.py'))

dct = set_parameters_for_clearmap(**params)
#resampling operations:
#######################
#resampling for the correction of stage movements during the acquisition between channels:
resampleData(**CorrectionResamplingParameterCfos);
resampleData(**CorrectionResamplingParameterAutoFluo);

#Downsampling for alignment to the Atlas:
resampleData(**RegistrationResamplingParameter);

#%%
#Alignment operations:
######################
#correction between channels:
resultDirectory  = alignData(**CorrectionAlignmentParameter);

#alignment to the Atlas:
resultDirectory  = alignData(**RegistrationAlignmentParameter);

#%%
#Cell detection:
################

print ('{}minutes to complete resize'.format((time.time() - start) / 60))
detectCells(**ImageProcessingParameter);
print ('{}minutes to complete'.format((time.time() - start) / 60))
#%%
#Filtering of the detected peaks:
#################################
#Loading the results:
points, intensities = io.readPoints(ImageProcessingParameter["sink"]);

#Thresholding: the threshold parameter is either intensity or size in voxel, depending on the chosen "row"
#row = (0,0) : peak intensity from the raw data
#row = (1,1) : peak intensity from the DoG filtered data
#row = (2,2) : peak intensity from the background subtracted data
#row = (3,3) : voxel size from the watershed
points, intensities = thresholdPoints(points, intensities, threshold = (20, 900), row = (3,3));
#points, intensities = thresholdPoints(points, intensities, threshold = (20, 900), row = (2,2));
io.writePoints(FilteredCellsFile, (points, intensities));


## Check Cell detection (For the testing phase only, remove when running on the full size dataset)
#######################
#import ClearMap.Visualization.Plot as plt
#pointSource= os.path.join(BaseDirectory, FilteredCellsFile[0]);
#data = plt.overlayPoints(cFosFile, pointSource, pointColor = None, **cFosFileRange);
#io.writeData(os.path.join(BaseDirectory, 'cells_check.tif'), data);


# Transform point coordinates
#############################
points = io.readPoints(CorrectionResamplingPointsParameter["pointSource"]);
points = resamplePoints(**CorrectionResamplingPointsParameter);
points = transformPoints(points, transformDirectory = CorrectionAlignmentParameter["resultDirectory"], indices = False, resultDirectory = None);
CorrectionResamplingPointsInverseParameter["pointSource"] = points;
points = resamplePointsInverse(**CorrectionResamplingPointsInverseParameter);
RegistrationResamplingPointParameter["pointSource"] = points;
points = resamplePoints(**RegistrationResamplingPointParameter);
points = transformPoints(points, transformDirectory = RegistrationAlignmentParameter["resultDirectory"], indices = False, resultDirectory = None);
io.writePoints(TransformedCellsFile, points);




# Heat map generation
#####################
points = io.readPoints(TransformedCellsFile)
intensities = io.readPoints(FilteredCellsFile[1])

#Without weigths:
vox = voxelize(points, AtlasFile, **voxelizeParameter);
if not isinstance(vox, basestring):
  io.writeData(os.path.join(BaseDirectory, 'cells_heatmap.tif'), vox.astype('int32'));

#With weigths from the intensity file (here raw intensity):
voxelizeParameter["weights"] = intensities[:,0].astype(float);
vox = voxelize(points, AtlasFile, **voxelizeParameter);
if not isinstance(vox, basestring):
  io.writeData(os.path.join(BaseDirectory, 'cells_heatmap_weighted.tif'), vox.astype('int32'));





#Table generation:
##################
#With integrated weigths from the intensity file (here raw intensity):
ids, counts = countPointsInRegions(points, labeledImage = AnnotationFile, intensities = intensities, intensityRow = 0);
table = numpy.zeros(ids.shape, dtype=[('id','int64'),('counts','f8'),('name', 'a256')])
table["id"] = ids;
table["counts"] = counts;
table["name"] = labelToName(ids);
io.writeTable(os.path.join(BaseDirectory, 'Annotated_counts_intensities.csv'), table);

#Without weigths (pure cell number):
ids, counts = countPointsInRegions(points, labeledImage = AnnotationFile, intensities = None);
table = numpy.zeros(ids.shape, dtype=[('id','int64'),('counts','f8'),('name', 'a256')])
table["id"] = ids;
table["counts"] = counts;
table["name"] = labelToName(ids);
io.writeTable(os.path.join(BaseDirectory, 'Annotated_counts.csv'), table);



#####################
#####################
#####################
#####################
#####################
#####################
