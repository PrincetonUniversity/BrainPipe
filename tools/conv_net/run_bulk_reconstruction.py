#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:11:59 2019

@author: wanglab
"""

from subprocess import check_output
import os

#function to run
def sp_call(call):
    """ command line call function """ 
    print(check_output(call, shell=True)) 
    return

def submit_post_processing(scratch_dir, tracing_fld, to_reconstruct = False):
    """ submit reconstruction en masse """

    if not to_reconstruct:
        to_reconstruct = [xx for xx in os.listdir(scratch_dir) if "reconstructed_array.npy"
                      not in os.listdir(os.path.join(scratch_dir, xx)) 
                      and "output_chnks" in os.listdir(os.path.join(scratch_dir, xx))]   
    #call
    for pth in to_reconstruct:
        call = "sbatch cnn_postprocess.sh {}".format(os.path.join(tracing_fld, pth))
        print(call)
        sp_call(call)
        
if __name__ == "__main__":

    scratch_dir = "/jukebox/scratch/zmd"
    tracing_fld = "/jukebox/wang/pisano/tracing_output/antero_4x"

    submit_post_processing(scratch_dir, tracing_fld) 
