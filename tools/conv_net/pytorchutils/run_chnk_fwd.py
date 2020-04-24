#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:53:51 2018

@author: wanglab
"""

import os, numpy as np, sys, time, shutil
import collections
import tifffile

import torch
from torch import sigmoid
import dataprovider3 as dp

import forward
import utils

def main(noeval, **args):

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    net = utils.create_network(**params)
    if not noeval:
        net.eval()

    utils.log_tagged_modules(params["modules_used"], params["log_dir"],
                             params["log_tag"], params["chkpt_num"])

    #lightsheet mods - input folder contains list of our "big" patches
    input_fld = os.path.join(params["data_dir"], "input_chnks") #set patches directory 
    sys.stdout.write("running inference on: \n{}\n".format(os.path.basename(params["data_dir"]))); sys.stdout.flush()   
    output_fld = os.path.join(params["data_dir"], "output_chnks") #set output directory 
    
    jobid = int(params["jobid"]) #set patch no. to run through cnn
    
    #find files that need to be processed
    fls = [os.path.join(input_fld, xx) for xx in os.listdir(input_fld)]; fls.sort()
    
    #select the file to process for this array job
    if jobid > len(fls)-1:
        sys.stdout.write("\njobid {} > number of files {}".format(jobid, len(fls))); sys.stdout.flush()    
    else:    
        start = time.time() 
        dset = fls[jobid]
        
        fs = make_forward_scanner(dset, **params)
        
        output = forward.forward(net, fs, params["scan_spec"], #runs forward pass
                                 activation=params["activation"])

        save_output(output, dset, output_fld, **params) #saves tif       
        fs._init() #clear out scanner
        
        sys.stdout.write("patch {}: {} min\n".format(jobid+1, round((time.time()-start)/60, 1))); sys.stdout.flush()

def fill_params(expt_name, chkpt_num, gpus, nobn, model_fname, dset_name, tag, jobid):

    params = {}

    #Model params
    params["in_spec"]     = dict(input=(1,20,32,32))
    params["output_spec"] = collections.OrderedDict(soma=(1,20,32,32))
    params["width"]       = [32, 40, 80]
    params["activation"]  = sigmoid
    params["chkpt_num"]   = chkpt_num

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]   = expt_name
    params["expt_dir"]    = "/tigress/zmd/3dunet_data/ctb/network/{}".format(expt_name)
    params["model_dir"]   = os.path.join(params["expt_dir"], "models")
    params["log_dir"]     = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]     = os.path.join(params["expt_dir"], "forward")
    params["log_tag"]     = "fwd_" + tag if len(tag) > 0 else "fwd"
    params["output_tag"]  = tag
    params["jobid"]       = jobid

    #Dataset params
    params["data_dir"]    = "/scratch/gpfs/zmd/{}".format(dset_name)
#    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    params["dsets"]       = dset_name
    params["input_spec"]  = collections.OrderedDict(input=(20,192,192)) #dp dataset spec
    params["scan_spec"]   = collections.OrderedDict(soma=(1,20,192,192))
    params["scan_params"] = dict(stride=(0.75,0.75,0.75), blend="bump")

    #Use-specific Module imports
    params["model_class"] = utils.load_source(model_fname).Model

    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]   = [params["in_spec"], params["output_spec"],
                              params["width"]]
    params["model_kwargs"] = {}

    #Modules used for record-keeping
    params["modules_used"] = [__file__, model_fname, "layers.py"]

    return params


def make_forward_scanner(dset_name, data_dir, input_spec,
                         scan_spec, scan_params, **params):
    """ Creates a DataProvider ForwardScanner from a dset name """

    # Reading chunk of lightsheet tif
    img = tifffile.imread(dset_name)
    
    img = (img / 255.).astype("float32")

    # Creating DataProvider Dataset
    vd = dp.Dataset()

    vd.add_data(key="input", data=img)
    vd.set_spec(input_spec)

    # Returning DataProvider ForwardScanner
    return dp.ForwardScanner(vd, scan_spec, **scan_params)

def save_output(output, dset, output_fld, output_tag, jobid, chkpt_num, **params):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data:

        output_data = output.outputs.get_data(k)
        
        if len(output_tag) == 0:
            basename = "patch_{}_{}_{}.tif".format(str(jobid).zfill(10), k, chkpt_num)
        else:
            basename = "patch_{}_{}_{}_{}.tif".format(str(jobid).zfill(10), k, 
                                               chkpt_num, output_tag)

        full_fname = os.path.join(output_fld, basename)
        
        tifffile.imsave(full_fname, output_data[0,:,:,:], compress = 1)

    return full_fname

#============================================================



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("model_fname",
                        help="Model Template Name")
    parser.add_argument("chkpt_num", type=int,
                        help="Checkpoint Number")
    parser.add_argument("dset_name",
                        help="Inference Dataset Name")
    parser.add_argument("--nobn", action="store_true",
                        help="Whether net uses batch normalization")
    parser.add_argument("--gpus", default=["0"], nargs="+")
    parser.add_argument("--noeval", action="store_true",
                        help="Whether to use eval version of network")
    parser.add_argument("--tag", default="",
                        help="Output (and Log) Filename Tag")
    parser.add_argument("jobid", type=int,
                        help="Array Task ID corresponding to patch number")

    args = parser.parse_args()

    main(**vars(args))
