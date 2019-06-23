#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:20:45 2018

@author: tpisano
"""

import cv2
import skvideo.io
from skimage.external import tifffile
import skimage
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    input_video = 'path/to/input.avi'
    output_video = 'path/to/output.avi'
    fps = 10
    codec = 0
    
    vc = cv2.VideoCapture(input_video)
    valid,fr = vc.read()
    vw = cv2.VideoWriter(output_video, codec, fps, (fr.shape[1], fr.shape[0]))
    
    while valid:
        valid,fr = vc.read()
        #here you have a frame from origin, called fr, saving out to destination in the next line
        vw.write(fr)

    #mpeg
    src = '/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/injection/20170915_ymaze_cfos15/elastix/20170915_ymaze_cfos15_488_555_015na_1hfsds_z5um_150msec_resized_ch01/result.tif'
    dst = '/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/injection/qc/20170915_ymaze_cfos15.mp4'
    mpeg_writer(src, dst)

def mpeg_writer(src, dst, equalize = True):
    '''http://www.scikit-video.org/stable/io.html
     pip install scikit-video
     import skvideo.io
    
     rate = '25/1'     
     inputdict={'-r': rate,},outputdict={'-vcodec': 'libx264','-pix_fmt': 'yuv420p','-r': rate,}


    '''
    if type(src) == str: src = tifffile.imread(src).astype('uint16')
    if equalize: 
        src = src*1.0
        src = skimage.exposure.adjust_gamma(src, gamma=.5, gain=2)
        
    #convert to RGB
    src = np.stack([src, src, src],axis=-1).astype(np.uint8)
    
    skvideo.io.vwrite(dst, src) #
    return

'''
class matplotlib_3d_video:
    #from https://stackoverflow.com/questions/18344934/animate-a-rotating-3d-graph-in-matplotlib
    
    #pnts : N,3 numpy array
    #dst = '/home/wanglab/Downloads/tmp.mp4'
    #z,y,x

    
    def __init__(self, pnts, dst, fps=30):
        self.pnts = np.asarray(pnts)
        self.dst = dst
        self.fps = fps
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.scatter(self.pnts[:,2], self.pnts[:,1], self.pnts[:,02], marker='o', s=20, c="goldenrod", alpha=0.6)
        self.animated=False    
    def animate(self, i):
        self.anim = animation.FuncAnimation(self.fig, self.ax.view_init(elev=10., azim=i), init_func=self.ax, frames=360, interval=20, blit=True)
        self.ax.view_init(elev=10., azim=i)
    def save(self):
        if self.animate==False: animate(self)
        #self.
        
    
    
    # Animate
    #anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
    # Save
    #anim.save(dst, fps=fps, extra_args=['-vcodec', 'libx264'])
    
    


class matplotlib_3d_video:
    #from https://stackoverflow.com/questions/18344934/animate-a-rotating-3d-graph-in-matplotlib
    
    #pnts : N,3 numpy array
    #dst = '/home/wanglab/Downloads/tmp.mp4'
    #z,y,x
    #notufncitla
   
    def __init__(self, pnts, dst, fps=30):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = pnts[:,2]
        y = pnts[:,1]
        z = pnts[:,0]
        anim = animation.FuncAnimation(fig, animate, init_func=init,frames=360, interval=20, blit=True)
        # Save
        anim.save('/home/wanglab/Downloads/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

def init():
    ax.scatter(x,y,z, marker='o', s=20, c="goldenrod", alpha=0.6)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

'''
if False:
  # Create some random data, I took this piece from here:
  # http://matplotlib.org/mpl_examples/mplot3d/scatter3d_demo.py
  def randrange(n, vmin, vmax):
      return (vmax - vmin) * np.random.rand(n) + vmin
  n = 100
  xx = randrange(n, 23, 32)
  yy = randrange(n, 0, 100)
  zz = randrange(n, -50, -25)
  
  # Create a figure and a 3D Axes
  fig = plt.figure()
  ax = Axes3D(fig)
  
  # Create an init function and the animate functions.
  # Both are explained in the tutorial. Since we are changing
  # the the elevation and azimuth and no objects are really
  # changed on the plot we don't have to return anything from
  # the init and animate function. (return value is explained
  # in the tutorial.
  def init():
      ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
      return fig,
  
  def animate(i):
      ax.view_init(elev=10., azim=i)
      return fig,
  
  # Animate
  anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=360, interval=20, blit=True)
  # Save
  anim.save('/home/wanglab/Downloads/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
