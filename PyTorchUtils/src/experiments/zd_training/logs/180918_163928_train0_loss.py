#!/usr/bin/env python2
__doc__ = """

Loss functions.

Nicholas Turner <nturner.cs@princeton.edu>, 2017
"""

import torch
from torch import nn
import numpy as np

class BinomialCrossEntropyWithLogits(nn.Module):
    """ 
    A version of BCE w/ logits with the ability to mask
    out regions of output
    """

    def __init__(self):

      nn.Module.__init__(self)

    def forward(self, pred, label, mask=None):

      #Need masking for this application
      # copied from PyTorch's github repo
      neg_abs = - pred.abs()
      err = pred.clamp(min=0) - pred * label + (1 + neg_abs.exp()).log()

      if mask is None:
        cost = err.sum() #/ np.prod(err.size())
      else:
        cost = (err * mask).sum() #/ mask.sum()

      return cost

