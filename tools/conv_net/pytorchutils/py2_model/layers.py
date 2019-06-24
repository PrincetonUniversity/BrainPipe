#!/usr/bin/env
__doc__ = """

Basic building block layers for constructing nets implemented in PyTorch.

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Based on a similar module by
Kisuk Lee <kisuklee@mit.edu>, 2016-2017
"""


import torch
import torch.nn as nn
from torch.nn import init
import math


def pad_size(ks, mode):

  assert mode in ["valid","same","full"]

  if mode == "valid":
    return (0,0,0)

  elif mode == "same":
    assert all([ x % 2 for x in ks ])
    return tuple( x // 2 for x in ks )

  elif mode == "full":
    return tuple( x - 1 for x in ks )


class Conv(nn.Module):
  """ Bare bones 3D convolution module w/ MSRA init """

  def __init__(self, D_in, D_out, ks, st, pd, bias=True):

    nn.Module.__init__(self)
    self.conv = nn.Conv3d(D_in, D_out, ks, st, pd, bias=bias)
    init.kaiming_normal_(self.conv.weight)
    if bias:
      init.constant_(self.conv.bias, 0)


  def forward(self, x):
    return self.conv(x)


class FactConv(nn.Module):
  """ Factorized 3D convolution using Conv"""

  def __init__(self, D_in, D_out, ks, st, pd, bias=True):

    nn.Module.__init__(self)
    if ks[0] > 1:
      self.factor = Conv(D_in, D_out, (1,ks[1],ks[2]),
                         (1,st[1],st[2]), (0,pd[1],pd[2]), bias=False)
      ks = (ks[0],1,1)
      st = (st[0],1,1)
      pd = (pd[0],0,0)

    else:
      self.factor = None

    self.conv = Conv(D_in, D_out, ks, st, pd, bias)


  def forward(self, x):

    if self.factor is not None:
      return self.conv(self.factor(x))
    else:
      return self.conv(x)


class ConvT(nn.Module):
  """ Bare Bones 3D ConvTranspose module w/ MSRA init """

  def __init__(self, D_in, D_out, ks, st, pd=(0,0,0), bias=True):

    nn.Module.__init__(self)
    self.conv = nn.ConvTranspose3d(D_in, D_out, ks, st, pd, bias=bias)
    init.kaiming_normal_(self.conv.weight)
    if bias:
      init.constant_(self.conv.bias, 0)


  def forward(self, x):
    return self.conv(x)


class ResizeConv(nn.Module):
    """ Upsampling followed by a Convolution """

    def __init__(self, D_in, D_out, ks, st, pd, bias=True, mode="nearest"):

        nn.Module.__init__(self)

        self.upsample = Upsample2D(scale_factor=2, mode=mode)
        self.conv = Conv(D_in, D_out, ks, st, pd, bias=bias)


    def forward(self, x):

        return self.conv(self.upsample(x))


class Upsample2D(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):

        nn.Module.__init__(self)

        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)

    def forward(self, x):

        #upsample in all dimensions, and undo the z upsampling
        return self.upsample(x)[:,:,::self.scale_factor,:,:]


class FactConvT(nn.Module):
  """ Factorized 3d ConvTranspose using ConvT """

  def __init__(self, D_in, D_out, ks, st, pd=(0,0,0), bias=True):

    nn.Module.__init__(self)
    if ks[0] > 1:
      self.factor = ConvT(D_in, D_out, (2,ks[1],ks[2]),
                          (1,st[1],st[2]), (0,pd[1],pd[2]), bias=False)
      ks = (ks[0],1,1)
      st = (st[0],1,1)
      pd = (pd[0],0,0)

    else:
      self.factor = None

    self.conv = ConvT(D_in, D_out, ks, st, pd, bias)


  def forward(self, x):

    if self.factor is not None:
      return self.conv(self.factor(x))
    else:
      return self.conv(x)
