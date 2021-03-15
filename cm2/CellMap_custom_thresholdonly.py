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

    #%% Initialize workspace
    import os, sys
    import tiffile as tif
    sys.path.append("/scratch/ejdennis/rat_BrainPipe/ClearMap2")
    from ClearMap.Environment import *  #analysis:ignore


    def joinup_all(directory,additional_path_list,filename_to_add):
        list_out = []
        for newpath in additional_path_list:
            list_out.append(os.path.join(directory,newpath,filename_to_add))

    def process_list_of_files(directory, additional_path_list, filename_to_add, source_threshold, size_tuple):
        list_of_files = joinup_all(directory,additional_path_list,filename_to_add)
        thresholds = {
            'source' : source_threshold,
            'size'   : size_tuple
        }
    for file in list_of_files:
        print('working on file {}'.format(file))
        filetosave= "{}_filtered.npy".format(file[0:-8])
        cells.filter_cells(source = file,
                            sink = filetosave,
                            thresholds=thresholds);

if __name__ == "__main__":

    #directories and files
    directory = '/scratch/ejdennis'
    smartspim_488_list = ['cm2_brains/a253/ch_488',
                          'cm2_brains/e142/ch_488',
                          'cm2_brains/j316/ch_488',
                          'cm2_brains/j317/ch_488']
    smartspim_642_list = ['cm2_brains/a253/ch_642',
                          'cm2_brains/e142/ch_642',
                          'cm2_brains/e143/ch_642',
                          'cm2_brains/e144/ch_642',
                          'cm2_brains/e153/ch_642',
                          'cm2_brains/h234/ch_642',
                          'cm2_brains/j316/ch_642',
                          'cm2_brains/j317/ch_642']
    lavision_ch00_list = ['lightsheet/z267/_ch00',
                          'lightsheet/z265/_ch00',
                          'lightsheet/z268/_ch00',
                          'lightsheet/z269/_ch00']
    filename_to_add = 'cells_raw.npy'


    process_list_of_files(directory, smartspim_488_list, filename_to_add, 3, (30,120))
    print('done with 488')
    process_list_of_files(directory, smartspim_642_list, filename_to_add, 3, (30,120))
    print('done with 642')
    process_list_of_files(directory, lavision_ch00_list, filename_to_add, 3, (30,120))
    print('done with lavision')
