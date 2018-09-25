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

    def __init__(self, datadir, dsets=[], 
                                mode="train", patchsz=(32, 256, 256)): #original: 18 x 160 x 160

      assert mode in ["train","val","test"]

      datadir = os.path.expanduser(datadir)
  
      volnames = ["input", "cell_label"]
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
      
      img = read_file(os.path.join(os.path.join(datadir, 'inputRawImages'), dset_name + "_inputRawImages.h5")) #+'/inputRawImages' added by zmd 20180917      
      #print '\n     Read raw file correctly!\n'
      
      lbl = read_file(os.path.join(os.path.join(datadir, 'inputLabelImages'), dset_name + "inputLabelImages-segmentation.h5")).astype("float32") #+'/inputInputImages' added by zmd 20180917
      #print '\n     Read label file correctly!\n'
      
      img = dp.transform.divideby(img, val=2000.0, dtype="float32")
      lbl[lbl != 0] = 1

      vd = dp.VolumeDataset()
      vd.add_raw_data(key="input",       data=img)
      vd.add_raw_data(key="cell_label",  data=lbl)
      
      vd.set_spec(spec)
      
      print '\n     Built dataset for {}'.format(dset_name)
      return vd


    def _aug(self, mode):

      aug = dp.Augmentor()
      if mode == "train":
        aug.append('misalign', max_trans=17.0)
      aug.append('missing', max_sec=5, mode='mix',random_color=True)
      aug.append('blur', max_sec=5, mode='mix')
      if mode == "train":
        aug.append('warp')
        aug.append('greyscale', mode='mix')
        aug.append('flip')
      return aug


    def _post(self):
      post = dp.Transformer()
      return post

