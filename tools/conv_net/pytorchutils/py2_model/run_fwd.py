#!/usr/bin/env python

import os, imp
import collections

import torch
from torch.nn import functional as F
import dataprovider3 as dp

import forward
import utils
import models


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

    for dset in params["dsets"]:
        print(dset)

        fs = make_forward_scanner(dset, **params)

        output = forward.forward(net, fs, params["scan_spec"],
                                 activation=params["activation"])

        save_output(output, dset, **params)


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
    params["data_dir"]    = "/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages"
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

    # Reading EM image
    img = utils.read_h5(os.path.join(data_dir, dset_name + "_inputRawImages.h5"))
    img = (img / 2000.).astype("float32")

    # Creating DataProvider Dataset
    vd = dp.Dataset()

    vd.add_data(key="input", data=img)
    vd.set_spec(input_spec)

    # Returning DataProvider ForwardScanner
    return dp.ForwardScanner(vd, scan_spec, params=scan_params)


def save_output(output, dset_name, chkpt_num, fwd_dir, output_tag, **params):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data:

        output_data = output.outputs.get_data(k)

        if len(output_tag) == 0:
            basename = "{}_{}_{}.h5".format(dset_name, k, chkpt_num)
        else:
            basename = "{}_{}_{}_{}.h5".format(dset_name, k, 
                                               chkpt_num, output_tag)

        full_fname = os.path.join( fwd_dir, basename )

        utils.write_h5(output_data[0,:,:,:], full_fname)


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
