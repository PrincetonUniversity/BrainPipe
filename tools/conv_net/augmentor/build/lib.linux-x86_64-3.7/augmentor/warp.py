from __future__ import print_function
import numpy as np

from .augment import Augment
from .warping import warping
from .geometry.box import Box


class Warp(Augment):
    """Warping by combining 5 types of linear transformations:

        1. Continuous rotation
        2. Shear
        3. Twist
        4. Scale
        5. Perspective stretch
    """
    def __init__(self, skip=0, **params):
        self.skip = np.clip(skip, 0, 1)
        self.params = dict(params)
        self.do_warp = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss
        self.do_warp = np.random.rand() > self.skip
        if not self.do_warp:
            return dict(spec)

        self.imgs = self._validate(spec, imgs)

        # Save original spec.
        self.spec = dict(spec)

        # Compute the largest image size.
        box = Box((0,0,0), (0,0,0))
        for k, v in spec.items():
            box = box.merge(Box((0,0,0), v[-3:]))
        maxsz = tuple(box.size())

        # Random warp parameters
        params = dict(self.params)
        params.update(kwargs)
        warp_params = warping.getWarpParams(maxsz, **params)
        self.size = tuple(x for x in warp_params[0])
        size_diff = tuple(x-y for x,y in zip(self.size, maxsz))
        self.rot     = warp_params[1]
        self.shear   = warp_params[2]
        self.scale   = warp_params[3]
        self.stretch = warp_params[4]
        self.twist   = warp_params[5]

        # DEBUG
        # print(warp_params)

        # Increase tensor size.
        ret = dict()
        for k, v in spec.items():
            if k in imgs:
                ret[k] = v[:-3] + self.size
            else:
                ret[k] = v[:-3] + tuple(x+y for x,y in zip(v[-3:], size_diff))
        return ret

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_warp:
            for k, v in sample.items():
                v = np.transpose(v, (1,0,2,3))
                if k in self.imgs:
                    v = warping.warp3d(v, self.spec[k][-3:],
                            self.rot, self.shear,
                            self.scale, self.stretch, self.twist
                        )
                else:
                    v = warping.warp3dLab(v, self.spec[k][-3:], self.size,
                            self.rot, self.shear,
                            self.scale, self.stretch, self.twist
                        )
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.transpose(v, (1,0,2,3)))
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs
