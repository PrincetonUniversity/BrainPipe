#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:23:00 2016

@author: wanglab
"""
import os, numpy as np, cv2, zipfile

# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
#            Tim D. Smith <git@tim-smith.us>, 2015
# License: MIT

from io import BytesIO
import zipfile

import numpy as np


def read_roi(fileobj):
    '''
    points = read_roi(fileobj)

    Read ImageJ's ROI format. Points are returned in a nx2 array. Each row
    is in [row, column] -- that is, (y,x) -- order.
    '''
    # This is based on:
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html

    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256

    class RoiType:
        POLYGON = 0
        RECT = 1
        OVAL = 2
        LINE = 3
        FREELINE = 4
        POLYLINE = 5
        NOROI = 6
        FREEHAND = 7
        TRACED = 8
        ANGLE = 9
        POINT = 10

    def get8():
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != b'Iout':
        raise ValueError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()

    if roi_type not in [RoiType.FREEHAND, RoiType.POLYGON, RoiType.RECT, RoiType.POINT]:
        raise NotImplementedError('roireader: ROI type %s not supported' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()
    x1 = getfloat()
    y1 = getfloat()
    x2 = getfloat()
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise NotImplementedError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if roi_type == RoiType.RECT:
        if options & SUB_PIXEL_RESOLUTION:
            return np.array(
                [[y1, x1], [y1, x1+x2], [y1+y2, x1+x2], [y1+y2, x1]],
                dtype=np.float32)
        else:
            return np.array(
                [[top, left], [top, right], [bottom, right], [bottom, left]],
                dtype=np.int16)

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
        fileobj.seek(4*n_coordinates, 1)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)

    points[:, 1] = [getc() for i in range(n_coordinates)]
    points[:, 0] = [getc() for i in range(n_coordinates)]

    if options & SUB_PIXEL_RESOLUTION == 0:
        points[:, 1] += left
        points[:, 0] += top

    return points


def read_roi_zip(fname, include_roi_name=False, verbose=True):
    '''Wrapper for reading zip files generated from ImageJ (FIJI)
    
    include_roi_name (optional) 
        if true: returns list of (roi_name, contour)
        roi_name=z,y,x
        useful for extracting z (NOTE: ImageJ has one-based numerics vs Python w zero-based numerics)
    
    
    '''
    
    import zipfile
    
    try:
        if not include_roi_name:
            with zipfile.ZipFile(fname) as zf:
                return [read_roi(zf.open(n)) for n in zf.namelist()]
                                                    
        if include_roi_name:
            with zipfile.ZipFile(fname) as zf:
                return [(n, read_roi(zf.open(n))) for n in zf.namelist()]
    
    #hack to try and keep 
    except ValueError:
        import sys
        if verbose: sys.stdout.write('***There was an error with ROI file {},\n\n...attempting to fix. You should check this result'.format(fname)); sys.stdout.flush()
        lst = []
        with zipfile.ZipFile(fname) as zf:
            for n in zf.namelist():
                if len( n[:-4].split('-')) == 3:
                     lst.append(n)
        return lst
                
                        
                                                