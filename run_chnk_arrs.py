#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:22:39 2018

@author: wanglab
"""

#for cluster
import os, sys
print os.getcwd()
sys.path.append('/mnt/bucket/labs/wang/zahra/lightsheet_copy')
sys.path.append('/jukebox/wang/zahra/lightsheet_copy')
<REST OF IMPORTS>

#job
#get jobids from SLURM or argv
print sys.argv
stepid = int(sys.argv[1])    
jobid = int(sys.argv[2]) 

#input folder contains list of our "big" patches
input_folder = <path>
output_folder = <path>
if not os.path.exists(output_folder): os.mkdir(output_folder)

#find files that need to be processed
fls = [os.path.join(input_folder, xx) for xx in os.listdir(input_folder)]; fls.sort()

#select the file to process for this batch job
if jobid > len(fls):
    #essentially kill job if too high - doing this to hopefully help with karma score although might not make a difference
    print('Jobid {} > number of files {}'.format(jobid, len(fls)))
    
else:    
    fl = fls[jobid]

    #name to save new file:
    output_fl = os.path.join(output_folder, os.path.basename(fl))
    
    #run inference on it
    <>
    
    #save it as output_fl
    from skimage.external import tifffile
    tifffile.imsave(output_fl, out, compress=1)
