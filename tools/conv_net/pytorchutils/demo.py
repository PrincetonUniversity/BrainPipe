#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:20:25 2018

@author: wanglab

"""

import os
import sys
import time
import numpy as np
import collections

import torch
import tensorboardX
import dataprovider3 as dp

import utils
import train
import loss

import tifffile

from torch.nn import functional as F
import forward


def main_train(**args):

    # args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params_train(**args)

    utils.set_gpus(params["gpus"])

    utils.make_required_dirs(**params)

    tstamp = utils.timestamp()
    utils.log_params(params, tstamp=tstamp)
    utils.log_tagged_modules(params["modules_used"],
                             params["log_dir"], "train",
                             chkpt_num=params["chkpt_num"],
                             tstamp=tstamp)

    start_training(**params)


def main_fwd(noeval, **args):

    # args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params_fwd(**args)

    utils.set_gpus(params["gpus"])

    net = utils.create_network(**params)
    if not noeval:
        net.eval()

    utils.log_tagged_modules(params["modules_used"], params["log_dir"],
                             params["log_tag"], params["chkpt_num"])

    # lightsheet mods - input folder contains list of our "big" patches
    input_fld = os.path.join(params["data_dir"], "input_patches")
    # set directory
    output_fld = os.path.join(params["data_dir"], "cnn_output")
    # set patches directory

    if not os.path.exists(output_fld):
        os.mkdir(output_fld)
    jobid = 0  # for demo only

    # find files that need to be processed
    fls = [os.path.join(input_fld, xx) for xx in os.listdir(input_fld)]
    fls.sort()

    # select the file to process for this batch job
    if jobid > len(fls):
        sys.stdout.write("\njobid {} > number of files {}\n".format(jobid, len(fls)))
        sys.stdout.flush()
    else:
        dset = fls[jobid]

        start = time.time()

        fs = make_forward_scanner(dset, **params)
        sys.stdout.write("\striding by: {}".format(fs.stride))
        sys.stdout.flush()

        output = forward.forward(net, fs, params["scan_spec"],  # runs forward pass
                                 activation=params["activation"])

        save_output(output, dset, output_fld, jobid,
                    params["output_tag"], params["chkpt_num"])  # saves tif
        fs._init()  # clear out scanner

    sys.stdout.write("\patch {}: {} min\n".format(jobid+1, round((time.time()-start)/60, 1)))
    sys.stdout.flush()


def fill_params_train(expt_name, batch_sz, gpus,
                      sampler_fname, model_fname, augmentor_fname, **args):

    params = {}

    # Model params
    params["in_spec"] = dict(input=(1, 20, 192, 192))
    params["output_spec"] = collections.OrderedDict(soma=(1, 20, 192, 192))
    params["width"] = [32, 40, 80]

    # Training procedure params
    params["max_iter"] = 51
    params["lr"] = 0.00001
    params["test_intv"] = 25
    params["test_iter"] = 10
    params["avgs_intv"] = 10
    params["chkpt_intv"] = 10
    params["warm_up"] = 5
    params["chkpt_num"] = 0
    params["batch_size"] = batch_sz

    # Sampling params
    print("the working directory is: {}\n".format(os.getcwd()))
    params["data_dir"] = os.path.join(os.path.dirname(os.getcwd()), 'demo')
    assert os.path.isdir(params["data_dir"]), "nonexistent data directory"

    params["train_sets"] = [
        "train"
    ]

    params["val_sets"] = [
        "val"
    ]

    params["patchsz"] = (20, 192, 192)
    params["sampler_spec"] = dict(input=params["patchsz"],
                                  soma_label=params["patchsz"])

    # GPUS
    params["gpus"] = gpus

    # IO/Record params
    params["expt_name"] = expt_name
    params["expt_dir"] = os.path.join(params["data_dir"], "experiments/{}".format(expt_name))

    params["model_dir"] = os.path.join(params["expt_dir"], "models")
    params["log_dir"] = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"] = os.path.join(params["expt_dir"], "forward")
    params["tb_train"] = os.path.join(params["expt_dir"], "tb/train")
    params["tb_val"] = os.path.join(params["expt_dir"], "tb/val")

    # Use-specific Module imports
    params["model_class"] = utils.load_source(model_fname).Model
    params["sampler_class"] = utils.load_source(sampler_fname).Sampler
    params["augmentor_constr"] = utils.load_source(augmentor_fname).get_augmentation

    # "Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"] = [params["in_spec"], params["output_spec"],
                            params["width"]]
    params["model_kwargs"] = {}

    # modules used for record-keeping
    params["modules_used"] = [__file__, model_fname, sampler_fname,
                              augmentor_fname, "loss.py"]

    return params


def fill_params_fwd(expt_name, chkpt_num, gpus,
                    nobn, model_fname, tag):

    params = {}

    # Model params
    params["in_spec"] = dict(input=(1, 20, 192, 192))
    params["output_spec"] = collections.OrderedDict(soma=(1, 20, 192, 192))
    params["width"] = [32, 40, 80]
    params["activation"] = F.sigmoid
    params["chkpt_num"] = chkpt_num

    # GPUS
    params["gpus"] = gpus

    # IO/Record params
    params["expt_name"] = expt_name
    params["expt_dir"] = os.path.join(os.path.join(os.path.dirname(
        os.getcwd()), 'demo'), "experiments/{}".format(expt_name))
    params["model_dir"] = os.path.join(params["expt_dir"], "models")
    params["log_dir"] = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"] = os.path.join(params["expt_dir"], "forward")
    params["log_tag"] = "fwd_" + tag if len(tag) > 0 else "fwd"
    params["output_tag"] = tag

    # Dataset params
    params["data_dir"] = os.path.join(os.path.dirname(os.getcwd()), 'demo')
    assert os.path.isdir(params["data_dir"]), "nonexistent data directory"
    params["input_spec"] = collections.OrderedDict(input=(20, 192, 192))  # dp dataset spec
    params["scan_spec"] = collections.OrderedDict(soma=(1, 20, 192, 192))
    params["scan_params"] = dict(stride=(0.75, 0.75, 0.75), blend="bump")

    # Use-specific Module imports
    params["model_class"] = utils.load_source(model_fname).Model

    # "Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"] = [params["in_spec"], params["output_spec"],
                            params["width"]]
    params["model_kwargs"] = {}

    # Modules used for record-keeping
    params["modules_used"] = [__file__, model_fname, "layers.py"]

    return params


def start_training(model_class, model_args, model_kwargs,
                   sampler_class, sampler_spec, augmentor_constr,
                   chkpt_num, lr, train_sets, val_sets, data_dir,
                   model_dir, log_dir, tb_train, tb_val,
                   **params):

    # PyTorch Model
    net = utils.create_network(model_class, model_args, model_kwargs)
    train_writer = tensorboardX.SummaryWriter(tb_train)
    val_writer = tensorboardX.SummaryWriter(tb_val)
    monitor = utils.LearningMonitor()

    # Loading model checkpoint (if applicable)
    if chkpt_num != 0:
        utils.load_chkpt(net, monitor, chkpt_num, model_dir, log_dir)

    # DataProvider Stuff
    train_aug = augmentor_constr(True)
    train_sampler = utils.AsyncSampler(sampler_class(data_dir, sampler_spec,
                                                     vols=train_sets,
                                                     mode="train",
                                                     aug=train_aug))

    val_aug = augmentor_constr(False)
    val_sampler = utils.AsyncSampler(sampler_class(data_dir, sampler_spec,
                                                   vols=val_sets,
                                                   mode="val",
                                                   aug=val_aug))

    loss_fn = loss.BinomialCrossEntropyWithLogits()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train.train(net, loss_fn, optimizer, train_sampler, val_sampler,
                train_writer=train_writer, val_writer=val_writer,
                last_iter=chkpt_num, model_dir=model_dir, log_dir=log_dir,
                monitor=monitor,
                **params)


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


def save_output(output, dset, output_fld, jobid, output_tag, chkpt_num):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data:

        output_data = output.outputs.get_data(k)

        if len(output_tag) == 0:
            basename = "{}_{}_{}.tif".format(str(jobid).zfill(10), k, chkpt_num)
        else:
            basename = "{}_{}_{}_{}.tif".format(str(jobid).zfill(10), k,
                                                chkpt_num, output_tag)

        full_fname = os.path.join(output_fld, basename)

        tifffile.imsave(full_fname, output_data[0, :, :, :], compress=1)

    return full_fname


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("model_fname",
                        help="Model Template filename")
    parser.add_argument("sampler_fname",
                        help="DataProvider Sampler filename")
    parser.add_argument("augmentor_fname",
                        help="Data Augmentor module filename")
    parser.add_argument("chkpt_num", type=int,
                        help="Checkpoint Number")
    parser.add_argument("--batch_sz",  type=int, default=1,
                        help="Batch size for each sample")
    parser.add_argument("--nobn", action="store_true",
                        help="Whether net uses batch normalization")
    parser.add_argument("--gpus", default=["0"], nargs="+")
    parser.add_argument("--noeval", action="store_true",
                        help="Whether to use eval version of network")
    parser.add_argument("--tag", default="",
                        help="Output (and Log) Filename Tag")

    args = parser.parse_args()

    main_train(**vars(args))

    del args.sampler_fname
    del args.augmentor_fname
    del args.batch_sz

    main_fwd(**vars(args))
