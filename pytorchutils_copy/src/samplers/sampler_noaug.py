#!/usr/bin/env python3
__doc__= """

Sampling class for joint synapse and vesicle training

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""
import os

import dataprovider as dp
import h5py

def read_file(fname):

    f = h5py.File(fname)
    d = f["/main"].value
    f.close()

    return d


class Sampler(object):

    def __init__(self, datadir, dsets=[], mode="train", patchsz=(18,160,160)):

      assert mode in ["train","val","test"]

      datadir = os.path.expanduser(datadir)
  
      volnames = ["input", "soma_label"]
      spec = { name : patchsz for name in volnames }

      self.dp = self.build_data_provider(datadir, spec, mode, dsets)


    def __call__(self, **kwargs):
      return self.dp("random", **kwargs)


    def build_data_provider(self, datadir, spec, mode, dsets):

      vdp = dp.VolumeDataProvider()

      for dset in dsets:
        vdp.add_dataset( self.build_dataset(datadir, spec, dset) )
      
      vdp.set_sampling_weights()

      vdp.set_augmentor(self._aug(mode))
      vdp.set_postprocessor(self._post())

      return vdp


    def build_dataset(self, datadir, spec, dset_name):

      print(dset_name)
      img = read_file(os.path.join(datadir, dset_name + "_image.h5"))      
      lbl = read_file(os.path.join(datadir, dset_name + "_somata.h5")).astype("float32")

      img = dp.transform.divideby(img, val=2000.0, dtype="float32")
      lbl[lbl != 0] = 1

      vd = dp.VolumeDataset()
      vd.add_raw_data(key="input",       data=img)
      vd.add_raw_data(key="soma_label",  data=lbl)
      
      vd.set_spec(spec)
      return vd


    def _aug(self, mode):

      aug = dp.Augmentor()
      if mode == "train":
        aug.append('flip')
      return aug


    def _post(self):
      post = dp.Transformer()
      return post

