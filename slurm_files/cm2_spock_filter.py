#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Jan 2021

@author: ejdennis
"""

import os
import sys
import tifffile as tif
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import matplotlib.colors

from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("/scratch/ejdennis/rat_BrainPipe/ClearMap2")
from ClearMap.Environment import sys, os, glob, np, plt, reload, settings, io, wsp, tfs, p3d, col, te, tmr, bp, ap, ano, res, elx, st, stw, clp, rnk, se, dif, skl, skp, vf, me, mr, vox, cells

import ClearMap.IO.Workspace as wsp

brainnames = ["j317","j316"]

for brain in brainnames:
#directories and files
    directory = '/scratch/ejdennis/cm2_brains/{}/ch_642'.format(brain)
    ws = wsp.Workspace('CellMap', directory=directory);
    thresholds = {
        'source' : 3,
        'size'   : (7,220)
        }
    cells.filter_cells(source = ws.filename('cells', postfix='raw'),
                       sink = ws.filename('cells', postfix='filtered'),
                       thresholds=thresholds);
print("done with brain {}".format(brain))
