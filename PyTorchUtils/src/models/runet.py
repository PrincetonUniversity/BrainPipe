#!/usr/bin/env python
__doc__ = """

3D U-Net.

(Optional)
Residual skip connections.

Kisuk Lee <kisuklee@mit.edu>, 2017-2018
Nicholas Turner <nturner@cs.princeton.edu>, 2017-2018
"""

import collections
from collections import OrderedDict
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


# Number of feature maps.
DEFAULT_NFEATURES = [16, 32, 64, 128, 256]


class RUNet(nn.Module):
    """Residual Symmetric U-Net (RSUNet).

    Args:
        in_channels (int): Input dimension.
        out_spec (dictionary): Output specification.
        depth (int): Depth/scale of U-Net.
        residual (bool, optional): Whether to use residual skip connections
        upsample (string, optional): Upsampling mode in
            ['bilinear', 'nearest', 'transpose']
        use_bn (bool, optional): Use batch normalization?
        momentum (float, optional): Momentum for batch normalization.

    Example:
        >>> in_channels  = 1
        >>> out_spec = {'affinity:12'}
        >>> model = RUNet(in_channels, out_spec, depth=4)
    """
    def __init__(self, in_channels, out_spec, depth,
                 residual=True, upsample='bilinear', use_bn=True,
                 momentum=0.001, embed_ks=(1,5,5), primary_ks=(3,3,3),
                 nfeatures=DEFAULT_NFEATURES, mode="same"):
        super(RUNet, self).__init__()
        #Flags
        self.residual = residual
        self.use_bn = use_bn
        self.mode = mode
        #Other params
        self.upsample = upsample
        self.momentum = momentum
        self.nfeatures = nfeatures
        self.out_spec = out_spec
        self.primary_ks = primary_ks

        embed_nin = embed_nout = self.nfeatures[0]

        # Model depth (# scales == depth + 1).
        assert depth < len(nfeatures), "depth >= specified feature levels"
        self.depth = depth

        # Input feature embedding without batchnorm.
        self.embed_in = EmbeddingMod(in_channels, embed_nin, embed_ks)
        in_channels = embed_nin

        # Contracting/downsampling pathway.
        for d in range(depth):
            fs = nfeatures[d]
            self.add_conv_mod(d, in_channels, fs)
            self.add_max_pool(d+1, fs)
            in_channels = fs

        # Bridge.
        fs = nfeatures[depth]
        self.add_conv_mod(depth, in_channels, fs)
        in_channels = fs

        # Expanding/upsampling pathway.
        for d in reversed(range(depth)):
            fs = nfeatures[d]
            self.add_upsample_mod(d, in_channels, fs)
            in_channels = fs
            self.add_expconv_mod(d, in_channels, fs)

        # Output feature embedding without batchnorm.
        self.embed_out = EmbeddingMod(in_channels, embed_nout, embed_ks)
        in_channels = embed_nout

        # Output by spec.
        self.output = OutputMod(in_channels, self.out_spec)

        self.crops = self.compute_crops()

    def add_conv_mod(self, depth, in_channels, out_channels):
        name = 'convmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels,
                         primary_kernel_size=self.primary_ks,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum, mode=self.mode)
        self.add_module(name, module)

    def add_expconv_mod(self, depth, in_channels, out_channels):
        """Expanding pathway convmod"""
        name = 'expconvmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels,
                         primary_kernel_size=self.primary_ks,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum, mode=self.mode)
        self.add_module(name, module)

    def add_max_pool(self, depth, in_channels, down=(1,2,2)):
        name = 'maxpool{}'.format(depth)
        module = nn.MaxPool3d(down)
        self.add_module(name, module)

    def add_upsample_mod(self, depth, in_channels, out_channels, up=(1,2,2)):
        name = 'upsample{}'.format(depth)
        module = UpsampleMod(in_channels, out_channels, up=up,
                             upsampling=self.upsample, use_bn=self.use_bn,
                             momentum=self.momentum, mode=self.mode)
        self.add_module(name, module)

    def compute_crops(self):
        """Computes 3d crop margins for each hierarchy depth"""
        crops = [0] * self.depth
        prev_crop = (0,0,0)

        for d in reversed(range(self.depth)):
            expconvmod = getattr(self, "expconvmod{}".format(d))
            new_crop = expconvmod.full_crop_margin()

            crop_z = (new_crop[0]+prev_crop[0],)
            #accounting for max pooling
            crop_yx = tuple(2*(new_crop[i]+prev_crop[i]) for i in range(1,3))

            crops[d] = crop_z + crop_yx
            prev_crop = new_crop

        return crops

    def forward(self, x):
        # Input feature embedding without batchnorm.
        x = self.embed_in(x)

        # Contracting/downsmapling pathway.
        skip = []
        for d in range(self.depth):
            print(d)
            convmod = getattr(self, 'convmod{}'.format(d))
            maxpool = getattr(self, 'maxpool{}'.format(d+1))
            x = convmod(x)
            skip.append(x)
            x = maxpool(x)

        # Bridge.
        bridge = getattr(self, 'convmod{}'.format(self.depth))
        x = bridge(x)

        # Expanding/upsampling pathway.
        for d in reversed(range(self.depth)):
            print(d)
            upsample = getattr(self, 'upsample{}'.format(d))
            expconvmod = getattr(self, 'expconvmod{}'.format(d))
            x = expconvmod(upsample(x, skip[d], self.crops[d]))

        # Output feature embedding without batchnorm.
        x = self.embed_out(x)
        return self.output(x)


def _ntuple(n):
    """
    Copied from PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad


def batchnorm(out_channels, use_bn, momentum=0.001):
    if use_bn:
        layer = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=momentum)
    else:
        layer = lambda x: x
    return layer


def residual_sum(x, skip, margin, residual):
    return x + crop3d(skip, margin) if residual else x


def crop3d(x, margin):
    shape = x.size()
    index3d = tuple(slice(b,e-b) for (b,e) in zip(margin,shape[-3:]))
    index = tuple(slice(0,e) for e in shape[:-3]) + index3d
    return x[index]


class Conv(nn.Module):
    """
    3D convolution w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal(self.conv.weight)
        if bias:
            nn.init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvT(nn.Module):
    """
    3D convolution transpose w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, bias=True):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        init.kaiming_normal(self.conv.weight)
        if bias:
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvMod(nn.Module):
    """
    Convolution module.
    """
    def __init__(self, in_channels, out_channels, primary_kernel_size,
                 activation=F.elu, residual=True, use_bn=True,
                 momentum=0.001, mode="same"):
        super(ConvMod, self).__init__()
        assert (in_channels == out_channels or
                in_channels == out_channels // 2), "improper dimensions"
        self.mode = mode
        self.dupl = in_channels == out_channels // 2
        # Convolution params.
        self.kszs = self.kernel_sizes(primary_kernel_size)
        ps = self.pad_sizes()
        self.crops = self.residual_crop_margins()
        bias = not use_bn
        # Convolutions.
        self.conv1 = Conv(in_channels,  out_channels, self.kszs[0], 1, ps[0], bias)
        self.conv2 = Conv(out_channels, out_channels, self.kszs[1], 1, ps[1], bias)
        self.conv3 = Conv(out_channels, out_channels, self.kszs[2], 1, ps[2], bias)
        self.conv4 = Conv(out_channels, out_channels, self.kszs[3], 1, ps[3], bias)
        # BatchNorm.
        self.bn1 = batchnorm(out_channels, use_bn, momentum=momentum)
        self.bn2 = batchnorm(out_channels, use_bn, momentum=momentum)
        self.bn3 = batchnorm(out_channels, use_bn, momentum=momentum)
        self.bn4 = batchnorm(out_channels, use_bn, momentum=momentum)
        # Activation function.
        self.activation = activation
        # Residual skip connection.
        self.residual = residual

    def pad_sizes(self, kszs=None, mode=None):
        kernel_sizes = self.kszs if kszs is None else kszs
        mode = self.mode if mode is None else mode
        return [pad_size(ks, mode) for ks in kernel_sizes]

    def kernel_sizes(self, primary_kernel_size):
        ks = _triple(primary_kernel_size)
        ks1 = (1,ks[1],ks[2])
        return (ks1,ks1,ks,ks)

    def residual_crop_margins(self, kszs=None, mode=None):
        """Amount to crop the residual"""
        kernel_sizes = self.kszs if kszs is None else kszs
        mode = self.mode if mode is None else mode
        return (self.full_crop_margin(kernel_sizes[0:2], mode),
                self.full_crop_margin(kernel_sizes[2:], mode))

    def full_crop_margin(self, kszs=None, mode=None):
        """Amount to crop in order to match the entire module"""
        kernel_sizes = self.kszs if kszs is None else kszs
        mode = self.mode if mode is None else mode
        if mode == "same":
            return (0,0,0)
        elif mode == "valid":
            # margins = padding for "same" convolutions
            pads = self.pad_sizes(kernel_sizes, "same")
            return tuple(map(sum,zip(*pads)))
        else:
            raise Exception("convolution mode {} not supported".format(mode))

    def forward(self, x):
        skip = torch.cat((x,x),dim=1) if self.dupl else x
        # Conv 1.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        # Conv 2.
        x = self.conv2(x)
        x = residual_sum(x, skip, self.crops[0], self.residual)
        x = self.bn2(x)
        x = self.activation(x)
        skip = x
        # Conv 3.
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        # Conv 4.
        x = self.conv4(x)
        x = residual_sum(x, skip, self.crops[1], self.residual)
        x = self.bn4(x)
        return self.activation(x)


class UpsampleMod(nn.Module):
    """
    Transposed Convolution module.
    """
    def __init__(self, in_channels, out_channels, up=(1,2,2),
                 upsampling='bilinear', activation=F.elu, use_bn=True,
                 momentum=0.001, mode="same"):
        super(UpsampleMod, self).__init__()
        # Convolution params.
        ks = (1,1,1)
        st = (1,1,1)
        pad = (0,0,0)
        bias = True
        # Upsampling.
        if upsampling == 'bilinear':
            self.up = nn.Upsample(scale_factor=up, mode='trilinear')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif upsampling == 'nearest':
            self.up = nn.Upsample(scale_factor=up, mode='nearest')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif upsampling == 'transpose':
            self.up = ConvT(in_channels, out_channels,
                            kernel_size=up, stride=up, bias=bias)
            self.conv = lambda x: x
        else:
            assert False, "unknown upsampling mode {}".format(mode)
        # BatchNorm and activation.
        self.bn = batchnorm(out_channels, use_bn, momentum=momentum)
        self.activation = activation

    def forward(self, x, skip, margin=(0,0,0)):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(residual_sum(x, skip, margin, True))
        return self.activation(x)


class EmbeddingMod(nn.Module):
    """
    Embedding module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu):
        super(EmbeddingMod, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                         stride=1, padding=pad, bias=True)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class OutputMod(nn.Module):
    """
    Embedding -> output module.

    Args:
        in_channels (int): Input dimension
        out_spec (dictionary): Output specification.
        kernel_size (int or 3-tuple, optional)
    """
    def __init__(self, in_channels, out_spec, kernel_size=1):
        super(OutputMod, self).__init__()

        self.spec = out_spec
        for (name, out_channels) in self.spec.items():
            conv = Conv(in_channels, out_channels, kernel_size, bias=True)
            setattr(self, name, conv)

    def forward(self, x):
        """
        Return an output list as "DataParallel" cannot handle an output
        dictionary.
        """
        return [getattr(self, k)(x) for k in self.spec]

Model = RUNet
