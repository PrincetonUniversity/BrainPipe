#!/usr/bin/env python
__doc__ = """

Training Script

Put all the ugly things that change with every experiment here

Nicholas Turner, 2017-8
"""

import os, imp
import collections

import torch
import tensorboardX

import utils
import train
import loss


def main(**args):

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    utils.make_required_dirs(**params)

    tstamp = utils.timestamp()
    utils.log_params(params, tstamp=tstamp)
    utils.log_tagged_modules(params["modules_used"],
                             params["log_dir"], "train",
                             chkpt_num=params["chkpt_num"],
                             tstamp=tstamp)

    start_training(**params)


def fill_params(expt_name, chkpt_num, batch_sz, gpus,
                sampler_fname, model_fname, augmentor_fname, **args):

    params = {}

    #Model params
    params["in_spec"]	   = dict(input=(1,20,32,32))
    params["output_spec"]  = collections.OrderedDict(cleft=(1,20,32,32))
    params["width"]        = [32, 40, 80]

    #Training procedure params
    params["max_iter"]    = 1000000
    params["lr"]          = 0.00001
    params["test_intv"]   = 100
    params["test_iter"]   = 10
    params["avgs_intv"]   = 50
    params["chkpt_intv"]  = 1000
    params["warm_up"]     = 50
    params["chkpt_num"]   = chkpt_num
    params["batch_size"]  = batch_sz

    #Sampling params
    params["data_dir"]     = "/home/wanglab/Documents/cfos_inputs/otsu_and_guassian_screened"
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    
    params["train_sets"] = ['dp_ann_201812_pcdev_lob6_4_forebrain_cortex_z200-219',
                              'tp_ann_201812_pcdev_lob6_9_forebrain_hypothal_z520-539',
                             'tp_ann_201812_pcdev_crus1_23_forebrain_cortex_z290-309',
                             'jd_ann_201904_an19_ymazefos_020719_thal_z350-369',
                             'jd_ann_201904_an21_ymazefos_020719_hypothal_z450-469',
                             'dp_ann_201904_an19_ymazefos_020719_pfc_z380-399',
                             'dp_ann_201904_an21_ymazefos_020719_hypothal_z450-469',
                             'tp_ann_201904_an10_ymzefos_020719_cortex_z280-279',
                             'jd_ann_201904_an22_ymazefos_020719_pfc_z150-169',
                             'jd_ann_201904_an22_ymazefos_020719_cb_z160-179',
                             'dp_ann_201904_an22_ymazefos_020719_cb_z160-179',
                             'tp_ann_201904_an19_ymazefos_020719_pfc_z380-399',
                             'dp_ann_201904_an12_ymazefos_020719_hypothal_z420-449',
                             'tp_ann_201812_pcdev_crus1_23_forebrain_midbrain_z260-279',
                             'tp_ann_201904_an4_ymazefos_020119_cortex_z200-219',
                             'tp_ann_201904_an4_ymazefos_020119_pfc_z200-219',
                             'tp_ann_201812_pcdev_lob6_4_forebrain_cortex_z200-219',
                             'dp_ann_201904_an19_ymazefos_020719_cortex_z380-399_02',
                             'tp_ann_201904_an22_ymazefos_020719_pfc_z150-169',
                             'jd_ann_201904_an30_ymazefos_020719_pfc_z410-429',
                             'jd_ann_201904_an10_ymazefos_020719_hypothal_z460-479',
                             'jd_ann_201904_an10_ymazefos_020719_pb_z260-279',
                             'jd_ann_201904_an30_ymazefos_020719_cortex_z400-419',
                             'dp_ann_201904_an19_ymazefos_020719_cortex_z350-369',
                             'dp_ann_an16_ymazecfos_z260-299_retrosplenial_cropped']

    params["val_sets"] = ['dp_ann_201904_an12_ymazefos_020719_cortex_z371-390',
                         'dp_ann_201904_an19_ymazefos_020719_cb_z380-399',
                         'dp_ann_201812_pcdev_lob6_9_forebrain_hypothal_z520-539',
                         'jd_ann_201904_an30_ymazefos_020719_striatum_z416-435',
                         'tp_ann_201904_an30_ymazefos_020719_striatum_z416-435',
                         'dp_ann_an22_ymazecfos_z230-249_sm_cortex_cropped',
                         'dp_ann_201904_an19_ymazefos_020719_thal_z350-369']
                    
                    
    params["patchsz"]	   = (20,32,32)
    params["sampler_spec"] = dict(input=params["patchsz"],
                                  soma_label=params["patchsz"])

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]  = expt_name
    params["expt_dir"]   = "/home/wanglab/Documents/cfos_net/experiment_dirs/{}".format(expt_name)

    params["model_dir"]  = os.path.join(params["expt_dir"], "models")
    params["log_dir"]    = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]    = os.path.join(params["expt_dir"], "forward")
    params["tb_train"]   = os.path.join(params["expt_dir"], "tb/train")
    params["tb_val"]     = os.path.join(params["expt_dir"], "tb/val")

    #Use-specific Module imports
    params["model_class"]   = utils.load_source(model_fname).Model
    params["sampler_class"] = utils.load_source(sampler_fname).Sampler
    params["augmentor_constr"] = utils.load_source(augmentor_fname).get_augmentation

    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]     = [params["in_spec"], params["output_spec"],
                                params["width"]]
    params["model_kwargs"]   = {}

    #modules used for record-keeping
    params["modules_used"] = [__file__, model_fname, sampler_fname,
                              augmentor_fname, "loss.py"]

    return params


def start_training(model_class, model_args, model_kwargs,
                   sampler_class, sampler_spec, augmentor_constr,
                   chkpt_num, lr, train_sets, val_sets, data_dir,
                   model_dir, log_dir, tb_train, tb_val,
                   **params):

    #PyTorch Model
    net = utils.create_network(model_class, model_args, model_kwargs)
    train_writer = tensorboardX.SummaryWriter(tb_train)
    val_writer = tensorboardX.SummaryWriter(tb_val)
    monitor = utils.LearningMonitor()

    #Loading model checkpoint (if applicable)
    if chkpt_num != 0:
        utils.load_chkpt(net, monitor, chkpt_num, model_dir, log_dir)

    #DataProvider Stuff
    train_aug = augmentor_constr(True)
    train_sampler = utils.AsyncSampler(sampler_class(data_dir, sampler_spec,
                                                     vols=train_sets,
                                                     mode="train",
                                                     aug=train_aug))

    val_aug = augmentor_constr(False)
    val_sampler   = utils.AsyncSampler(sampler_class(data_dir, sampler_spec,
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description= __doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("model_fname",
                        help="Model Template filename")
    parser.add_argument("sampler_fname",
                        help="DataProvider Sampler filename")
    parser.add_argument("augmentor_fname",
                        help="Data Augmentor module filename")
    parser.add_argument("--batch_sz",  type=int, default=1,
                        help="Batch size for each sample")
    parser.add_argument("--chkpt_num", type=int, default=0,
                        help="Checkpoint Number")
    parser.add_argument("--gpus", default=["0"], nargs="+")

    args = parser.parse_args()


    main(**vars(args))

