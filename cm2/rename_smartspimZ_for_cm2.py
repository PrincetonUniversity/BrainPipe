#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created Jan 2021

@author: ejdennis
"""

import os,sys, glob,shutil
import numpy as np
import matplotlib.pyplot as plt
import sys

# symlink files to a new directory with filenames Z0000.tif, Z0001.tif, etc...
# Change the dst_dir to where you want these z planes linked

print("in rename, source is")
print(sys.argv[1])
print("in rename, destination is")
print(sys.argv[2])

src_dir = sys.argv[1]

# dst_dir is where you want your new labeled Zs saved
dst_dir = sys.argv[2]
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

print("dst dir is {}".format(dst_dir))

src_files = sorted(glob.glob(src_dir + '/**/*.tif', recursive=True))
print("src files")
print(src_files)

for ii,src in enumerate(src_files):
    dst_basename = 'Z' + f'{ii}'.zfill(4) + '.tif'
    dst = os.path.join(dst_dir,dst_basename)
    os.symlink(src,dst)
print("in python done with files")
