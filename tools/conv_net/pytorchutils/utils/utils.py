#!/usr/bin/env python
__doc__ = """

Miscellaneous Utils

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import importlib
import datetime
import shutil
import types
import os
import re

import torch
from torch.autograd import Variable
import numpy as np
import h5py, tifffile


__all__ = ["timestamp",
           "make_required_dirs","log_tagged_modules","log_params",
           "create_network","load_network","load_learning_monitor",
           "save_chkpt","load_chkpt","iter_from_chkpt_fname",
           "load_source",
           "to_torch", "masks_empty",
           "read_img","write_img",
           "set_gpus"]


def timestamp():
    return datetime.datetime.now().strftime("%d%m%y_%H%M%S")


def make_required_dirs(model_dir, log_dir, fwd_dir, tb_train, tb_val, **params):

    for d in [model_dir, log_dir, fwd_dir, tb_train, tb_val]:
        if not os.path.isdir(d):
            os.makedirs(d)


def log_params(param_dict, tstamp=None, log_dir=None):

    if log_dir is None:
        assert "log_dir" in param_dict, "log dir not specified"
        log_dir = param_dict["log_dir"]

    tstamp = tstamp if tstamp is not None else timestamp()

    output_basename = "{}_params.csv".format(tstamp)

    with open(os.path.join(log_dir,output_basename), "w+") as f:
        for (k,v) in param_dict.items():
            f.write("{k};{v}\n".format(k=k,v=v))


def log_tagged_modules(module_fnames, log_dir, phase, chkpt_num=0, tstamp=None):

    tstamp = tstamp if tstamp is not None else timestamp()

    for fname in module_fnames:
        basename = os.path.basename(fname)
        output_basename = "{}_{}{}_{}".format(tstamp, phase, chkpt_num, basename)

        shutil.copyfile(fname, os.path.join(log_dir, output_basename))


def save_chkpt(model, learning_monitor, chkpt_num, model_dir, log_dir):

    # Save model
    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(chkpt_num))
    torch.save(model.module.state_dict(), chkpt_fname)

    # Save learning monitor
    lm_fname = os.path.join(log_dir, "stats{}.h5".format(chkpt_num))
    learning_monitor.save(lm_fname, chkpt_num)


def create_network(model_class, model_args, model_kwargs,
                   chkpt_num=0, model_dir=None, **params):

    net = torch.nn.DataParallel(model_class(*model_args, **model_kwargs)).cuda()

    if chkpt_num > 0:
        load_network(net, chkpt_num, model_dir)

    return net


def load_network(model, chkpt_num, model_dir):

    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(chkpt_num))
 #   model.load_state_dict(torch.load(chkpt_fname)) use if loading python 2 model - zmd
    model.module.load_state_dict(torch.load(chkpt_fname))

    return model


def load_learning_monitor(learning_monitor, chkpt_num, log_dir):

    lm_fname = os.path.join(log_dir, "stats{}.h5".format(chkpt_num))
    learning_monitor.load(lm_fname)

    return learning_monitor


def load_chkpt(model, learning_monitor, chkpt_num, model_dir, log_dir):

    m = load_network(model, chkpt_num, model_dir)

    lm = load_learning_monitor(learning_monitor, chkpt_num, log_dir)

    return m, lm


def load_source(fname, module_name="module"):
    """Updated version of imp.load_source(fname)"""
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def iter_from_chkpt_fname(chkpt_fname):
    """ Extracts the iteration number from a network checkpoint """
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def to_torch(np_arr, block=True):
    tensor = torch.from_numpy(np.ascontiguousarray(np_arr))
    return tensor.cuda(non_blocking=(not block))


def masks_empty(sample, mask_names):
    """ Tests whether a sample has any non-masked values """
    return any(not np.any(sample[name]) for name in mask_names)


def set_gpus(gpu_list):
    """ Sets the gpus visible to this process """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)


def read_img(fname):
    
    if fname[-3:] == ".h5":
        with h5py.File(fname) as f:
            d = f["/main"].value
    elif fname[-4:] == ".tif":
        d = tifffile.imread(fname)
    else:
        raise RuntimeError("only hdf5 and tiff format is supported")
        
    return d


def write_img(data, fname):

    if os.path.exists(fname):
      os.remove(fname)

    if fname[-3:] == ".h5":
        with h5py.File(fname) as f:
            f.create_dataset("/main",data=data)
    elif fname[-4:] == ".tif":
        tifffile.imsave(fname, data)
    else:
        raise RuntimeError("only hdf5 and tiff format is supported")
