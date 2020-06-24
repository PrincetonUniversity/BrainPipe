import numpy as np
import time

from ..geometry import *
from ..tensor import WritableTensorData as WTD
from ..tensor import WritableTensorDataWithMask as WTDM


def prepare_outputs(spec, locs, blend=False, blend_mode=''):
    if blend_mode not in ['','bump']:
        raise RuntimeError('unknown output blend type [%s]' % blend_mode)
    if blend_mode == 'bump':
        outputs = BumpBlend(spec, locs, blend=blend)
    else:
        outputs = Blend(spec, locs, blend=blend)
    return outputs


class Blend(object):
    """
    Blend interface.
    """
    def __init__(self, spec, locs, blend=False):
        self.spec = dict(spec)
        self.locs = list(locs)
        self.blend = blend
        self._prepare_data()

    def push(self, loc, sample):
        for k, v in sample.items():
            assert(k in self.data)
            self.data[k].set_patch(loc, v, op=self.op)

    def get_data(self, key):
        assert(key in self.data)
        return self.data[key].data()

    def voxels(self):
        voxels = list()
        for k, v in self.data.items():
            voxels.append(np.prod(v.dim()))
        return min(voxels) if len(voxels) > 0 else 0

    ####################################################################
    ## Private Methods.
    ####################################################################

    def _prepare_data(self):
        assert(len(self.locs) > 0)
        lmin = self.locs[0]
        lmax = self.locs[-1]

        self.data = dict()
        self.op = None
        for k, v in self.spec.items():
            dim = v[-3:]
            a = centered_box(lmin, dim)
            b = centered_box(lmax, dim)
            c = a.merge(b)
            shape = v[:-3] + tuple(c.size())

            # Inference with overlapping windows
            if self.blend:
                self.data[k] = WTDM(shape, c.min())
                self.op = np.add
            else:
                self.data[k] = WTD(shape, c.min())


class BumpBlend(Blend):
    """
    Blending with a bump function.
    """
    def __init__(self, spec, locs, blend=False):
        super(BumpBlend, self).__init__(spec, locs, blend=blend)
        self.logit_maps = dict()
        self.max_logits = None
        if blend:
            # Precompute maximum of overlapping bump logits
            # for numerical stability.
            max_logits = dict()
            for k, v in self.data.items():
                fov = self.spec[k][-3:]
                data = np.full(v.dim(), -np.inf, dtype='float32')
                max_logit = WTD(data, offset=v.offset())
                max_logit_window = self._bump_logit_map(fov)
                for loc in self.locs:
                    max_logit.set_patch(loc, max_logit_window, op=np.maximum)
                max_logits[k] = max_logit
            self.max_logits = max_logits

    def push(self, loc, sample):
        for k, v in sample.items():
            assert(k in self.data)
            t0 = time.time()
            mask = self._get_mask(k, loc, v.shape[-3:])
            t1 = time.time()
            self.data[k].set_patch(loc, v, op=self.op, mask=mask)
            t2 = time.time()
            # print("get_mask: %.3f, set_patch: %.3f" % (t1 - t0, t2 - t1))

    ####################################################################
    ## Private methods.
    ####################################################################

    def _get_mask(self, key, loc, dim):
        mask = None
        if self.blend:
            assert(key in self.max_logits)
            max_logit = self.max_logits[key].get_patch(loc, dim)
            mask = self._bump_map(max_logit.shape[-3:], max_logit[0,...])
        return mask

    def _bump_logit(self, z, y, x, t=1.5):
        return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)

    def _bump_logit_map(self, dim):
        ret = self.logit_maps.get(dim)
        if ret is None:
            x = range(dim[-1])
            y = range(dim[-2])
            z = range(dim[-3])
            zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
            xv = (xv + 1.0)/(dim[-1] + 1.0)
            yv = (yv + 1.0)/(dim[-2] + 1.0)
            zv = (zv + 1.0)/(dim[-3] + 1.0)
            ret = self._bump_logit(zv, yv, xv)
            self.logit_maps[dim] = ret
        return ret

    def _bump_map(self, dim, max_logit):
        return np.exp(self._bump_logit_map(dim) - max_logit)
