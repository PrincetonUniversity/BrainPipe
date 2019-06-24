from __future__ import print_function
from itertools import repeat
import collections
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ['Model']


WIDTH = [16,32,64,128,256,512]


def _ntuple(n):
    """
    Copied from the PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        return _triple(0)
    elif mode == 'same':
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == 'full':
        return tuple(x - 1 for x in ks)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                       bias=False):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class BilinearUp(nn.Module):
    """Caffe style bilinear upsampling.
    Currently everything's hardcoded and only supports upsampling factor of 2.
    """
    def __init__(self, in_channels, out_channels):
        super(BilinearUp, self).__init__()
        assert in_channels==out_channels
        self.groups = in_channels
        self.init_weights()

    def forward(self, x):
        return F.conv_transpose3d(x, self.weight,
            stride=(1,2,2), padding=(0,1,1), groups=self.groups
        )

    def init_weights(self):
        weight = torch.Tensor(self.groups, 1, 1, 4, 4)
        width = weight.size(-1)
        hight = weight.size(-2)
        assert width==hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
        self.register_buffer('weight', weight)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    padding = pad_size(kernel_size, 'same')
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BNReLUConv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', conv(in_channels, out_channels,
                                     kernel_size=kernel_size))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = BNReLUConv(channels, channels)
        self.conv2 = BNReLUConv(channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.add_module('pre',  BNReLUConv(in_channels, out_channels))
        self.add_module('res',  ResBlock(out_channels))
        self.add_module('post', BNReLUConv(out_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=up, mode='trilinear'),
            BilinearUp(in_channels, in_channels),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x, skip):
        return self.up(x) + skip


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=(1,2,2)):
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, up=up)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x, skip)
        return self.conv(x)


class RSUNet(nn.Module):
    def __init__(self, width=WIDTH):
        super(RSUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1

        self.in_channels = width[0]

        self.iconv = ConvBlock(width[0], width[0])

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(nn.Sequential(nn.MaxPool3d((1,2,2)),
                                             ConvBlock(width[d], width[d+1])))
        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(UpConvBlock(width[d+1], width[d]))

        self.out_channels = width[0]

        self.init_weights()

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        for uconv in self.uconvs:
            x = uconv(x, skip.pop())

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        spec = collections.OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))
        outs = []
        for k, v in spec.items():
            out_channels = v[-4]
            outs.append(Conv(in_channels, out_channels, kernel_size, bias=True))
        self.outs = nn.ModuleList(outs)
        self.keys = spec.keys()

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        return [out(x) for k, out in zip(self.keys, self.outs)]


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, in_spec, out_spec, width=WIDTH):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = list(in_spec.values())[0][-4]
        out_channels = width[0] #matches the RSUNet output
        io_kernel = (1,5,5)

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', RSUNet(width=width))
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel))