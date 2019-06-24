#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:00:03 2018

@author: wanglab
"""

import os

import h5py, tifffile

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset


class Sampler(object):

    def __init__(self, datadir, spec, vols=[], mode="train", aug=None):
        assert mode in ["train","val"], "invalid mode:{}".format(mode)
        datadir = os.path.expanduser(datadir)
        self.build(datadir, vols, spec, aug)

    def __call__(self):
        return self.dataprovider()

    def build(self, datadir, vols, spec, aug):
        print("Spec")
        print(spec)
        dp = DataProvider(spec)
        for vol in vols:
            print("Vol: {}".format(vol))
            dp.add_dataset(self.build_dataset(datadir, vol))
        dp.set_augment(aug)
        dp.set_imgs(["input"])
        dp.set_segs(["soma_label"])
        self.dataprovider = dp

    def build_dataset(self, datadir, vol):
        
        # Reading either hdf5 or tif training data; raw image has to be consistent with label image
        if os.path.isfile(os.path.join(datadir, vol + "_img.h5")):
            img = read_img(os.path.join(datadir, vol + "_img.h5"))
            soma = read_img(os.path.join(datadir, vol + "_lbl.h5")).astype("float32")
        elif os.path.isfile(os.path.join(datadir, vol + "_img.tif")):
            img = read_img(os.path.join(datadir, vol + "_img.tif"))
            soma = read_img(os.path.join(datadir, vol + "_lbl.tif")).astype("float32")
        
        #Preprocessing
        img = (img / 255.).astype("float32")
        soma[soma != 0] = 1

        # Create Dataset.
        dset = Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='soma_label', data=soma)
        return dset


    
def read_img(fname):
    
    """ by zmd """
    
    assert os.path.isfile(fname)
    
    if fname[-3:] == ".h5":
        with h5py.File(fname) as f:
            d = f["/main"].value
    elif fname[-4:] == ".tif":
        d = tifffile.imread(fname)
    else:
        raise RuntimeError("only hdf5 and tiff format is supported")
        
    return d

