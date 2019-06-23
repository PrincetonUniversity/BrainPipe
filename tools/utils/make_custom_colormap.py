#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:42:51 2018

@author: tpisano
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize



class MidpointNormalize(Normalize):
    '''
       from tools.utils.make_custom_colormap import MidpointNormalize
       my_cmap = plt.cm.RdBu
       norm = MidpointNormalize(vmin=minn,vmax=maxx,midpoint=0)
       my_cmap.set_under('white',alpha=0)
       fig = plt.imshow(np.ma.masked_values(tarr[z],0), cmap=my_cmap, vmax=maxx, vmin=minn, norm=norm)
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y),0)




def make_custom_colormap(value_colormap_dct):
    '''
colordf = pd.read_pickle('/home/wanglab/wang/pisano/figures/deformation_based_geometry/v3/data/colordf.p')
    dict(zip(colordf.soi_vol_val,colordf.Color))
    
	eseentially need a linear colormap

{1.0: (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 2.0: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 3.0: (1.0, 0.7333333333333333, 0.47058823529411764),
 4.0: (1.0, 0.4980392156862745, 0.054901960784313725),
 5.0: (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 6.0: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 7.0: (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 8.0: (1.0, 0.596078431372549, 0.5882352941176471),
 9.0: (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 10.0: (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 11.0: (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 12.0: (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 13.0: (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)}

import seaborn as sns
dct = {i:sns.color_palette('tab20')[i] for i in range(20)}
cmap = make_custom_colormap(dct)

    '''
    
    
    interval = 0.5
    
    #add 0 for white
    cmap = ListedColormap([(1.0,1.0,1.0)]+value_colormap_dct.values())
    vals = value_colormap_dct.keys()
    vals.sort()
    bounds = [1,vals[0] - interval]
    for i in range(len(vals)-1):
        delta = abs(vals[i] - vals[i+1]) / 2.0
        bounds.append(vals[i+1]-delta)
    bounds.append(vals[-1]+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
     
    return norm, cmap


#%%
def imagej_colormap(dct, dst):
    '''import seaborn as sns
    dct = {i+1:sns.color_palette('tab20')[i] for i in range(20)}
    dst = '/jukebox/wang/pisano/figures/deformation_based_geometry/v3/analysis/sois/soi_vol_lut.lut'

#####################################
	#adjusting for olf = 2 and iso ==2
    dct = {i+1:sns.color_palette('tab20')[i] for i in range(20)}
    dst = '/jukebox/wang/pisano/figures/deformation_based_geometry/v3/analysis/sois/soi_vol_lut_olf_iso_correct.lut'
    
    #generate volume - to save - 3d surface viewer   
    #generate fiji lut #0-255, r,g,b values
    lines = ['{} {} {}\n'.format(i,0,0) for i in range(256)]
    for v,c in {int(v):[int(cc*255) for cc in c] for v,c in dct.iteritems()}.iteritems():
        lines[v] = '{} {} {}\n'.format(c[0], c[1], c[2])
    #flipping
    lines[1], lines[2] = lines[2], lines[1]
    lines = ''.join(lines)
    with open(dst, 'w+') as fl:
        fl.write(lines)
        fl.close()
    shutil.copy(dst, '/home/wanglab/Fiji.app/luts')
#########################################



    '''
    #generate volume - to save - 3d surface viewer   
    #generate fiji lut #0-255, r,g,b values
    lines = ['{} {} {}\n'.format(i,0,0) for i in range(256)]
    for v,c in {int(v):[int(cc*255) for cc in c] for v,c in dct.iteritems()}.iteritems():
        lines[v] = '{} {} {}\n'.format(c[0], c[1], c[2])
    lines = ''.join(lines)
    with open(dst, 'w+') as fl:
        fl.write(lines)
        fl.close()
    shutil.copy(dst, '/home/wanglab/Fiji.app/luts')
    #need hard 8bit with it
    #soi_vol = tifffile.imread('/jukebox/wang/pisano/figures/deformation_based_geometry/v3/data/soi_vol.tif').astype('uint8')
    #tifffile.imsave('/jukebox/wang/pisano/figures/deformation_based_geometry/v3/data/soi_vol_8bit.tif', soi_vol)
    #generate volume - to save - 3d surface viewer
    #soi_vol = tifffile.imread('/jukebox/wang/pisano/figures/deformation_based_geometry/v3/data/soi_vol_8bit.tif')
    #sitk.Show(sitk.GetImageFromArray(soi_vol[:270]))
    #saved in '/jukebox/wang/pisano/figures/deformation_based_geometry/v3/analysis/sois/
    
