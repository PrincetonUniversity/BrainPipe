#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 20200830

@author: ejdennis
"""

import pandas as pd
import numpy as np
import sys
import tifffile as tif
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe")

affinefile = "/home/emilyjanedennis/Desktop/mouseann_to_rat_AFFINE.tif"
finalwarpfile = "/home/emilyjanedennis/Desktop/mouseann_to_rat.tif"

affine = tif.imread(affinefile)
finalwarp = tif.imread(finalwarpfile)

annotationfile = "/home/emilyjanedennis/Desktop/for_registration_to_lightsheet/PMA_ann.csv"
anns = pd.read_csv(annotationfile, usecols=[0, 1], names=['id', 'abbr'])

anns.insert(2, 'affine_vox', 0)
anns.insert(3, 'mouserat_vox', 0)

for i in range(0, np.size(anns['id'])-1):
    anns.mouserat_vox[i] = np.size(finalwarp[finalwarp == anns.id[i]])
    anns.affine_vox[i] = np.size(affine[affine == anns.id[i]])

anns['change'] = anns['affine_vox']-anns['mouserat_vox']
anns = anns[anns.mouserat_vox > 0]
annspercent = anns['change']/anns['mouserat_vox']*100
anns.percent[np.isinf(anns.percent)] = 0
anns.percent.replace(0, np.nan)

anns.to_csv(r"/home/emilyjanedennis/Desktop/PMA_ann_mouserat_vox_percent.csv")
finalwarp_relative = finalwarp
ids = anns.id
uniquevals = np.unique(finalwarp)
finalonlyval = np.setdiff1d(uniquevals, ids)

for i in ids:
    finalwarp_relative[finalwarp_relative == i] = 0

for i in ids:
    finalwarp_relative[finalwarp_relative == i] = anns.loc[
        anns['id'] == i, 'percent'].iloc[0]
