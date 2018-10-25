#!/usr/bin/env python
__doc__ = """

Miscellaneous Utils

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import torch
from torch.autograd import Variable
import numpy as np

import os, re, shutil
import datetime
import h5py


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
    torch.save(model.state_dict(), chkpt_fname)

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
    model.load_state_dict(torch.load(chkpt_fname))

    return model


def load_learning_monitor(learning_monitor, chkpt_num, log_dir):

    lm_fname = os.path.join(log_dir, "stats{}.h5".format(chkpt_num))
    learning_monitor.load(lm_fname)

    return learning_monitor


def load_chkpt(model, learning_monitor, chkpt_num, model_dir, log_dir):

    m = load_network(model, chkpt_num, model_dir)

    lm = load_learning_monitor(learning_monitor, chkpt_num, log_dir)

    return m, lm


def iter_from_chkpt_fname(chkpt_fname):
    """ Extracts the iteration number from a network checkpoint """
    basename = os.path.basename(chkpt_fname)
    return int(re.findall(r"\d+", basename)[0])


def masks_empty(sample, mask_names):
    """ Tests whether a sample has any non-masked values """
    return any(not np.any(sample[name]) for name in mask_names)


def make_variable(np_arr, requires_grad=True, volatile=False):
    """ Creates a torch.autograd.Variable from a np array """
    if not volatile:
      return Variable(torch.from_numpy(np_arr), requires_grad=requires_grad).cuda()
    else:
      return Variable(torch.from_numpy(np_arr), volatile=True).cuda()


def set_gpus(gpu_list):
    """ Sets the gpus visible to this process """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)


def read_h5(fname):

    with h5py.File(fname) as f:
        d = f['/main'].value

    return d


def write_h5(data, fname):

    if os.path.exists(fname):
      os.remove(fname)

    with h5py.File(fname) as f:
        f.create_dataset("/main",data=data)
