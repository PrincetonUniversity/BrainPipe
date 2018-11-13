#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:58:38 2018

@author: wanglab
"""

import os, numpy as np, sys, time, gc
import collections

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

    sys.stdout.write('\n      Using torch version: {}\n\n'.format(torch.__version__)) #check torch version is correct - use 0.4.1

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    net = utils.create_network(**params)
    if not noeval:
        net.eval()

    utils.log_tagged_modules(params["modules_used"], params["log_dir"],
                             params["log_tag"], params["chkpt_num"])

    #lightsheet mods
    inputs = load_memmap_arr(os.path.join(params["data_dir"], "patched_memmap_array.npy")) #load input patched array 
    output_arr = load_memmap_arr(os.path.join(params["data_dir"], 'patched_prediction_array.npy'), mode = 'w+', dtype = 'float32', shape = inputs.shape) #initialise output probability map
    
    initial = time.time()
    
    for i in range(inputs.shape[0]): #iterates through each large patch to run inference #len(inputs[0])       
               
        start = time.time()
        
        dset = inputs[i,:,:,:] #grabs chunk
                        
        fs = make_forward_scanner(dset, **params) #makes scanner
                
        output = forward.forward(net, fs, params["scan_spec"], #runs forward pass
                                 activation=params["activation"])

        output_arr[i,:,:,:] = save_output(output, output_arr[i,:,:,:], **params) #saves probability array
                   
        if i%5==0: output_arr.flush() #flush out output array to harddrive
        fs._init() #clear out scanner
        
        sys.stdout.write("\nPatch {}: {} min".format((i+1), round((time.time()-start)/60, 1))); sys.stdout.flush()

    sys.stdout.write("\n***************************************************************************************\
                     \nTotal time spent predicting: {} hrs\n".format(round((time.time()-initial)/3600, 0))); sys.stdout.flush()


def fill_params(expt_name, chkpt_num, gpus,
                nobn, model_name, dset_names, tag):

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
    params["expt_dir"]    = "/jukebox/wang/zahra/conv_net/training/experiment_dirs/{}".format(expt_name)
    params["model_dir"]   = os.path.join(params["expt_dir"], "models")
    params["log_dir"]     = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]     = os.path.join(params["expt_dir"], "forward")
    params["log_tag"]     = "fwd_" + tag if len(tag) > 0 else "fwd"
    params["output_tag"]  = tag

    #Dataset params
    params["data_dir"]    = "/jukebox/scratch/20180327_jg42_bl6_lob6a_05"
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    params["dsets"]       = dset_names
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

    # Reading chunk of lightsheet memory mapped array
    img = (dset_name / 255.).astype("float32")

    # Creating DataProvider Dataset
    vd = dp.Dataset()

    vd.add_data(key="input", data=img)
    vd.set_spec(input_spec)

    # Returning DataProvider ForwardScanner
    return dp.ForwardScanner(vd, scan_spec, params=scan_params)
    del img, vd

def save_output(output, output_arr, **params):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data:

        output_data = output.outputs.get_data(k)

        output_arr = output_data[0,:,:,:]

    return output_arr

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
    parser.add_argument("dset_names", nargs="+",
                        help="Inference Dataset Names")
    parser.add_argument("--nobn", action="store_true",
                        help="Whether net uses batch normalization")
    parser.add_argument("--gpus", default=["0"], nargs="+")
    parser.add_argument("--noeval", action="store_true",
                        help="Whether to use eval version of network")
    parser.add_argument("--tag", default="",
                        help="Output (and Log) Filename Tag")


    args = parser.parse_args()

    main(**vars(args))

