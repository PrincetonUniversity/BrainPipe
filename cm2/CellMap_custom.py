#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellMap
=======

This script is the main pipeline to analyze immediate early gene expression
data from iDISCO+ cleared tissue [Renier2016]_.

See the :ref:`CellMap tutorial </CellMap.ipynb>` for a tutorial and usage.


.. image:: ../Static/cell_abstract_2016.jpg
   :target: https://doi.org/10.1016/j.cell.2020.01.028
   :width: 300

.. figure:: ../Static/CellMap_pipeline.png

  iDISCO+ and ClearMap: A Pipeline for Cell Detection, Registration, and
  Mapping in Intact Samples Using Light Sheet Microscopy.


References
----------
.. [Renier2016] `Mapping of brain activity by automated volume analysis of immediate early genes. Renier* N, Adams* EL, Kirst* C, Wu* Z, et al. Cell. 2016 165(7):1789-802 <https://doi.org/10.1016/j.cell.2016.05.007>`_
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

if __name__ == "__main__":

  #%%############################################################################
  ### Initialization
  ###############################################################################

  #%% Initialize workspace
  import os, sys
  sys.path.append("/home/emilyjanedennis/Desktop/GitHub/ClearMap2")
  from ClearMap.Environment import *  #analysis:ignore

  #directories and files
  directory = '/home/emilyjanedennis/Desktop/clearmap/j317/'   
  
  expression_raw      = '071880_094731_<Z,6>.tif'           
  expression_auto      = '071880_094731_<Z,6>.tif'           
  
  ws = wsp.Workspace('CellMap', directory=directory);
  ws.update(raw=expression_raw, autofluorescence=expression_auto)
  ws.info()

  ws.debug = True
  resources_directory = settings.resources_path


  #%%############################################################################
  ### Data conversion
  ###############################################################################

  #%% Convet raw data to npy file

  print("converting raw data to npy")
  source = ws.source('raw');
  sink   = ws.filename('stitched')
  #io.delete_file(sink)
  #io.convert(source, sink, processes=None, verbose=True);
  #print("converted raw data to npy")

  #%%############################################################################
  ### Resampling and atlas alignment
  ###############################################################################

  #%% Resample

  #resample_parameter = {
#      "source_resolution" : (1.063, 1.063, 10),
#      "sink_resolution"   : (10,10,10),
#      "processes" : 4,
#      "verbose" : True,
#      };

  #io.delete_file(ws.filename('resampled'))
  #print("set resample params and deleted resampled file")
  #res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)
  #print("resampled")
  #%%############################################################################
  ### Cell detection
  ###############################################################################

  #%% Cell detection:

  cell_detection_parameter = cells.default_cell_detection_parameter.copy();
  cell_detection_parameter['illumination_correction'] = None;
  cell_detection_parameter['background_correction']['shape'] = (3,3,3);
  cell_detection_parameter['intensity_detection']['measure'] = ['source'];
  cell_detection_parameter['shape_detection']['threshold'] = 220;
  io.delete_file(ws.filename('cells', postfix='maxima'))
  print("finished setting cell detection params")
  
  slicing = (slice(5000,10000),slice(5000,10000),slice(2000,2400));
  ws.create_debug('stitched', slicing=slicing);

  #%%############################################################################
  ### Cell detection
  ###############################################################################

  #%% Cell detection:

  processing_parameter = cells.default_cell_detection_processing_parameter.copy();
  processing_parameter.update(
      processes = 'serial', # 'serial',
      size_max = 100, #100, #35,
      size_min = 10,  #30
      overlap  = 5, #32, #10,
      verbose=True)
  print("set processing params")

  cells.detect_cells('/home/emilyjanedennis/Desktop/clearmap/j317/debug_stitched.npy', ws.filename('cells', postfix='raw'),
                     cell_detection_parameter=cell_detection_parameter,
                     processing_parameter=processing_parameter)


  #%% Filter cells

  thresholds = {
      'source' : 3,
      'size'   : (10,120)
      }

  cells.filter_cells(source = ws.filename('cells', postfix='raw'),
                     sink = ws.filename('cells', postfix='filtered'),
                     thresholds=thresholds);
