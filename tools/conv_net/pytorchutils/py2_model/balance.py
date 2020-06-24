import torch
import numpy as np


def gunpowder_balance(labels, mask=None, thresh=0.):

    if mask is not None:
        bmsk = (mask > 0)
        nmsk = bmsk.sum().item()
        assert(nmsk > 0)
    else:
        bmsk = torch.ones_like(labels, dtype=torch.uint8)
        nmsk = np.prod(bmsk.size())

    lpos = (torch.gt(labels, thresh) * bmsk).type(torch.float)
    lneg = (torch.le(labels, thresh) * bmsk).type(torch.float)

    npos = lpos.sum().item()

    fpos = np.clip(npos / nmsk, 0.05, 0.95)
    fneg = (1.0 - fpos)

    wpos = 1. / (2. * fpos)
    wneg = 1. / (2. * fneg)

    return (lpos * wpos + lneg * wneg).type(torch.float32)
