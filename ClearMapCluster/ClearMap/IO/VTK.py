# -*- coding: utf-8 -*-
"""
Interface to write points to VTK files

Notes:
    - points are assumed to be in [x,y,z] coordinates as standard in ClearMap
    - reading of points not supported at the moment!

"""
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.
# Modified from a matlab code by Kannan Umadevi Venkataraju

#from evtk.hl import pointsToVTK 

import numpy;

import ClearMap.IO as io;

def writePoints(filename, points, labelImage = None):
    """Write point data to vtk file
    
    Arguments:
        filename (str): file name
        points (array): point data
        labelImage (str, array or None): optional label image to determine point label
    
    Returns:
        str: file name
    """

    #y = points[:,0];
    #x = points[:,1];
    #z = points[:,2];    
    
    x = points[:,0];
    y = points[:,1];
    z = points[:,2];    
    nPoint = x.size;
    
    #print nPoint;
    
    pointLabels = numpy.ones(nPoint);
    if not labelImage is None:
        if isinstance(labelImage, str):
            labelImage = io.readData(labelImage);
            
        dsize = labelImage.shape;
        for i in range(nPoint):
            #if y[i] >= 0 and y[i] < dsize[0] and x[i] >= 0 and x[i] < dsize[1] and z[i] >= 0 and z[i] < dsize[2]:
            if x[i] >= 0 and x[i] < dsize[0] and y[i] >= 0 and y[i] < dsize[1] and z[i] >= 0 and z[i] < dsize[2]:
                 #pointLabels[i] = labelImage[y[i], x[i], z[i]];
                 pointLabels[i] = labelImage[x[i], y[i], z[i]];
        
    #write VTK file
    vtkFile = open(filename, 'w')
    vtkFile.write('# vtk DataFile Version 2.0\n');
    vtkFile.write('Unstructured Grid Example\n');
    vtkFile.write('ASCII\n');
    vtkFile.write('DATASET UNSTRUCTURED_GRID\n');
    vtkFile.write("POINTS " + str(nPoint) + " float\n")
    for iPoint in range(nPoint):
        vtkFile.write(str(x[iPoint]).format('%05.20f') + " " +  str(y[iPoint]).format('%05.20f') + " " + str(z[iPoint]).format('%05.20f') + "\n");    
    
    vtkFile.write("CELLS " + str(nPoint) + " " + str(nPoint * 2) + "\n");


    for iPoint in range(nPoint):
        vtkFile.write("1 " + str(iPoint) + "\n");
    vtkFile.write("CELL_TYPES " + str(nPoint) + "\n");
    for iPoint in range(0, nPoint):
        vtkFile.write("1 \n");
    #vtkFile.write("\n");
    vtkFile.write("POINT_DATA " + str(nPoint) + "\n");
    vtkFile.write('SCALARS scalars float 1\n');
    vtkFile.write("LOOKUP_TABLE default\n");
    for iLabel in pointLabels:
        vtkFile.write(str(int(iLabel)) + " ");
        #vtkFile.write("1 ")
    vtkFile.write("\n");
    vtkFile.close();   
    
    return filename;


def readPoints(filename, **args):
    """Read points form vtk file
    
    Notes:
        - Not implmented yet !
    """
    raise RuntimeError('readPoints for VTK files not implmented!');

