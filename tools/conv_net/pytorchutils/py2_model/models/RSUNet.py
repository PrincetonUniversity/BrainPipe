#!/usr/bin/env python
__doc__ = """
Symmetric 3d U-Net implemented in PyTorch
(Optional)
Factorized 3D convolution
Extra residual connections
Nicholas Turner <nturner@cs.princeton.edu>, 2017
Based on an architecture by
Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import layers


#Global switches
factorize = False
residual  = True
bn = True

# Number of feature maps
nfeatures = [28,36,48,64,80,96]

# Filter size
sizes = [(1,3,3),
         (3,3,3),
         (3,3,3),
         (3,3,3),
         (3,3,3),
         (3,3,3)]

# In/out filter & stride size
io_size   = (1,5,5)
io_stride = (1,1,1)

class ConvMod(nn.Module):
  """ Convolution "module" """

  def __init__(self, D_in, D_out, ks, activation=F.elu,
               fact=factorize, resid=residual, 
               bn=True, momentum=0.001):
   
    nn.Module.__init__(self) 
    st = (1,1,1)
    pd = layers.pad_size(ks, "same")
    # conv layer constructor 
    conv_constr = layers.FactConv if fact else layers.Conv
    bias = not bn

    self.resid = resid
    self.bn = bn
    self.activation = activation

    first_pd = layers.pad_size((1,ks[1],ks[2]), "same")
    self.conv1 = conv_constr(D_in, D_out, (1,ks[1],ks[2]), st, first_pd, bias)
    self.conv2 = conv_constr(D_out, D_out, ks, st, pd, bias)
    self.conv3 = conv_constr(D_out, D_out, ks, st, pd, bias)

    if self.bn:
      self.bn1 = nn.BatchNorm3d(D_out, momentum=momentum)
      self.bn2 = nn.BatchNorm3d(D_out, momentum=momentum)
      self.bn3 = nn.BatchNorm3d(D_out, momentum=momentum)
    

  def forward(self, x):

    out1 = self.conv1(x)
    if self.bn:
      out1 = self.bn1(out1)
    out1 = self.activation(out1)

    out2 = self.conv2(out1)
    if self.bn:
      out2 = self.bn2(out2)
    out2 = self.activation(out2)

    out3 = self.conv3(out2)
    
    if self.resid:
      out3 = out3 + out1
   
    if self.bn:
      return self.activation(self.bn3(out3))
    else:
      return self.activation(out3)


class ConvTMod(nn.Module):
  """ Transposed Convolution "module" """

  def __init__(self, D_in, D_out, ks, up=(1,2,2), activation=F.elu,
               fact=factorize, resid=residual, 
               bn=True, momentum=0.001):

    nn.Module.__init__(self) 
    
    #ConvT constructor
    convt_constr = layers.FactConvT if fact else layers.ConvT
    self.bn = bn
    self.activation = activation
    bias = not bn

    self.convt = convt_constr(D_in, D_out, ks=up, st=up, bias=bias)
    if bn:
      self.bn1 = nn.BatchNorm3d(D_out, momentum=momentum)

    self.convmod = ConvMod(D_out, D_out, ks, fact=fact, resid=resid, bn=bn)


  def forward(self, x, skip):
    
    if self.bn:
      convt = self.activation(self.bn1(self.convt(x) + skip))
    else:
      convt = self.activation(self.convt(x) + skip)

    return self.convmod(convt)
    

class Conv(nn.Module):
  """ Single convolution module """

  def __init__(self, D_in, D_out, ks, st=(1,1,1), activation = F.elu,
               fact=factorize):
    
    nn.Module.__init__(self) 
    pd = layers.pad_size(ks, "same")
    
    conv_constr = layers.FactConv if fact else layers.Conv
    self.activation = activation

    self.conv = conv_constr(D_in, D_out, ks, st, pd, bias=True)


  def forward(self, x):
    return self.activation(self.conv(x))


class OutputModule(nn.Module):
  """ Hidden representation -> Output module """

  def __init__(self, D_in, outspec, ks=io_size, st=io_stride):
    """ outspec should be an Ordered Dict """

    nn.Module.__init__(self) 

    pd = layers.pad_size(ks, "same")

    self.output_layers = []
    for (name,d_out) in outspec.items():
      setattr(self, name, layers.Conv(D_in, d_out, ks, st, pd, bias=True))
      self.output_layers.append(name)


  def forward(self, x):
    return [ getattr(self,layer)(x) for layer in self.output_layers ]


class RSUNet(nn.Module):
  """ Full model """

  def __init__(self, D_in, output_spec, depth, io_size=io_size, io_stride=io_stride, bn=bn):

    nn.Module.__init__(self) 

    assert depth < len(nfeatures)
    self.depth = depth

    # D_in represents the input dimension (#feature maps) 
    # in most pytorch docs. I'll follow that convention here

    # Input feature embedding without batchnorm
    fs = nfeatures[0]
    self.inputconv = Conv(D_in, fs, io_size, st=io_stride)
    D_in = fs

    #modules within up/down pathways
    # are added with setattr calls
    # slightly obscured by U3D methods

    # Contracting pathway
    for d in range(depth):
      fs = nfeatures[d]
      ks = sizes[d]
      self.add_conv_mod(d, D_in, fs, ks, bn)
      self.add_max_pool(d+1, fs)
      D_in = fs

    # Bridge
    fs = nfeatures[depth]
    ks = sizes[depth]
    self.add_conv_mod(depth, D_in, fs, ks, bn)
    D_in = fs

    # Expanding pathway
    for d in reversed(range(depth)):
      fs = nfeatures[d]
      ks = sizes[d]
      self.add_deconv_mod(d, D_in, fs, bn, ks)
      D_in = fs

    # Output feature embedding without batchnorm
    self.embedconv = Conv(D_in, D_in, ks, st=(1,1,1))

    # Output by spec
    self.outputdeconv = OutputModule(D_in, output_spec, ks=io_size, st=io_stride)


  def add_conv_mod(self, depth, D_in, D_out, ks, bn):

    setattr(self, "convmod{}".format(depth),
            ConvMod(D_in, D_out, ks, bn=bn))


  def add_max_pool(self, depth, D_in, down=(1,2,2)):
    
    setattr(self, "maxpool{}".format(depth),
            nn.MaxPool3d(down))


  def add_deconv_mod(self, depth, D_in, D_out, bn, up=(1,2,2)):
    
    setattr(self, "deconv{}".format(depth),
            ConvTMod(D_in, D_out, up, bn=bn))


  def forward(self, x):

    # Input feature embedding without batchnorm
    x = self.inputconv(x)
    
    # Contracting pathway
    skip = []
    for d in range(self.depth):
      convmod = getattr(self, "convmod{}".format(d))
      maxpool = getattr(self, "maxpool{}".format(d+1))
      cd = convmod(x)
      x = maxpool(cd)

      skip.append(cd)

    # Bridge
    bridge = getattr(self, "convmod{}".format(self.depth))
    x = bridge(x)
    
    # Expanding pathway
    for d in reversed(range(self.depth)): 
      deconv = getattr(self, "deconv{}".format(d))
      x = deconv(x, skip[d])

    # Output feature embedding without batchnorm
    x = self.embedconv(x)

    return self.outputdeconv(x)

Model = RSUNet