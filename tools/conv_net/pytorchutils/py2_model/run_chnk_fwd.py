#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:22:39 2018

@author: wanglab
"""

import os, numpy as np, sys, time
import collections
import tifffile

import torch
from torch.nn import functional as F
import dataprovider3 as dp

import forward
import utils
import models

def load_memmap_arr(pth, mode='r', dtype = 'float32', shape = False):
    '''Function to load memmaped array.
    
    by @tpisano

    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

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
    input_fld = os.path.join(params["data_dir"], "patches") #set patches directory 
    output_fld = os.path.join(params["data_dir"], "cnn_patches") #set patches directory 
    
    if not os.path.exists(output_fld): os.mkdir(output_fld)
    jobid = int(params["jobid"]) #set patch no. to run through cnn
    
    #find files that need to be processed
    fls = [os.path.join(input_fld, xx) for xx in os.listdir(input_fld)]; fls.sort()
    
    #select the file to process for this batch job
    if jobid > len(fls):
        #essentially kill job if too high - doing this to hopefully help with karma score although might not make a difference
        sys.stdout.write("\nJobid {} > number of files {}\n".format(jobid, len(fls))); sys.stdout.flush()    
    else:    
        dset = fls[jobid]
        
        start = time.time()
        
        fs = make_forward_scanner(dset, **params)
        sys.stdout.write("\striding by: {}".format(fs.stride)); sys.stdout.flush()    
        
        output = forward.forward(net, fs, params["scan_spec"], #runs forward pass
                                 activation=params["activation"])

        save_output(output, dset, output_fld, **params) #saves tif       
        fs._init() #clear out scanner
        
    sys.stdout.write("\patch {}: {} min\n".format(jobid+1, round((time.time()-start)/60, 1))); sys.stdout.flush()


def fill_params(expt_name, chkpt_num, gpus,
                nobn, model_name, tag, jobid):

    params = {}

    #Model params
    params["in_dim"]      = 1
    params["output_spec"] = collections.OrderedDict(soma_label=1)
    params["depth"]       = 4
    params["batch_norm"]  = not(nobn)
    params["activation"]  = F.sigmoid
    params["chkpt_num"]   = chkpt_num

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]   = expt_name
    params["expt_dir"]    = "/tigress/zmd/3dunet_data/experiments/{}".format(expt_name)
    params["model_dir"]   = os.path.join(params["expt_dir"], "models")
    params["log_dir"]     = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]     = os.path.join(params["expt_dir"], "forward")
    params["log_tag"]     = "fwd_" + tag if len(tag) > 0 else "fwd"
    params["output_tag"]  = tag

    #Dataset params
    params["data_dir"]    = "/scratch/gpfs/zmd/20180327_jg40_bl6_sim_03"
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    params["jobid"]       = jobid
    params["input_spec"]  = collections.OrderedDict(input=(20,192,192)) #dp dataset spec
    params["scan_spec"]   = collections.OrderedDict(soma_label=(1,20,192,192))
    params["scan_params"] = dict(stride=(0.75,0.75,0.75), blend="bump")

    #Use-specific Module imports
    params["model_class"]  = utils.load_source('models/RSUNet.py').Model

    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]   = [params["in_dim"], params["output_spec"],
                             params["depth"] ]
    params["model_kwargs"] = { "bn" : params["batch_norm"] }

    #Modules used for record-keeping
    params["modules_used"] = [__file__, 'models/RSUNet.py', "layers.py"]

    return params


def make_forward_scanner(dset_name, data_dir, input_spec,
                         scan_spec, scan_params, **params):
    """ Creates a DataProvider ForwardScanner from a dset name """

    # Reading chunk of lightsheet tif
    img = tifffile.imread(dset_name)
    
    img = (img / 2000.).astype("float32") #2000 bc trained using cshl net

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
            basename = "{}_{}_{}.tif".format(jobid, k, chkpt_num)
        else:
            basename = "{}_{}_{}_{}.tif".format(jobid, k, 
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
    parser.add_argument("model_name",
                        help="Model Template Name")
    parser.add_argument("chkpt_num", type=int,
                        help="Checkpoint Number")
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
