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
  import tiffile as tif
  sys.path.append("/home/emilyjanedennis/Desktop/GitHub/ClearMap2")
  from ClearMap.Environment import *  #analysis:ignore

  #directories and files
  directory = '/home/emilyjanedennis/Desktop/brains/'
  listoffiles = ['/home/emilyjanedennis/Desktop/brains/z265/z265_cells_raw.npy',
                '/home/emilyjanedennis/Desktop/brains/z266/z266_debug_cells_raw.npy',
                '/home/emilyjanedennis/Desktop/brains/z267/debug_cells_raw.npy',
                '/home/emilyjanedennis/Desktop/brains/z268/_ch00_cells_raw.npy',
                '/home/emilyjanedennis/Desktop/brains/z268/_ch01_cells_raw.npy',
                '/home/emilyjanedennis/Desktop/brains/z269/_ch01_cells_raw.npy']
  thresholds = {
      'source' : 3,
      'size'   : (30,120)
      }
  for file in listoffiles:
      filetosave= "{}_filtered.npy".format(file[0:-8])
      cells.filter_cells(source = file,
                        sink = filetosave,
                        thresholds=thresholds);
