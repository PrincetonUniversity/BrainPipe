from __future__ import print_function
import numpy as np

from .augment import Augment, Compose
from .perturb import Perturb


class Section(Augment):
    """Perturb random sections in a training sample.

    Args:
        perturb (``Perturb``): ``Perturb`` class.
        maxsec (int):
        prob (float, optional):
        skip (float, optional): skip probability.
        double (bool, optional): double section.
    """
    def __init__(self, perturb, maxsec=0, prob=None, skip=0, double=False,
                 **params):
        assert issubclass(perturb, Perturb)
        self.perturb = perturb
        assert (maxsec > 0) or (prob is not None)
        self.maxsec = max(maxsec, 0)
        self.prob = np.clip(prob, 0, 1) if prob is not None else prob
        self.skip = np.clip(skip, 0, 1)
        self.margin = int(double)
        self.params = params
        self.zlocs = []
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss
        if np.random.rand() < self.skip:
            self.zlocs = []
            return dict(spec)

        # Random sections
        zdim = self._validate(spec, imgs) - self.margin
        if self.prob is None:
            nsecs = np.random.randint(1, int(self.maxsec) + 1)
            zlocs = np.random.choice(zdim, nsecs, replace=False)
        else:
            zlocs = np.random.rand(zdim) <= self.prob
            zlocs = np.where(zlocs)[0]
        self.zlocs = zlocs
        self.imgs = imgs
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for z in self.zlocs:
            if self.margin > 0:
                z = slice(z, z + self.margin + 1)
            perturb = self.get_perturb()
            for k in self.imgs:
                perturb(sample[k][...,z,:,:])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'perturb={}, '.format(self.perturb)
        if self.prob is None:
            format_string += 'maxsec={}, '.format(self.maxsec)
        else:
            format_string += 'prob={:.2f}, '.format(self.prob)
        format_string += 'skip={:.2f}, '.format(self.skip)
        format_string += 'double={}, '.format(bool(self.margin))
        format_string += 'params={}'.format(self.params)
        format_string += ')'
        return format_string

    def get_perturb(self):
        return self.perturb(**self.params)

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        zdims = [spec[k][-3] for k in imgs]
        zmin, zmax = min(zdims), max(zdims)
        assert zmax==zmin  # Do not allow inputs with different z-dim.
        assert zmax>self.maxsec
        return zmax


class PartialSection(Section):
    def get_perturb(self):
        class _PerturbQuadrant():
            def __init__(self, perturb, rx, ry, quad):
                self.perturb = perturb
                self.rx = rx
                self.ry = ry
                self.quad = quad

            def __call__(self, img):
                x = int(np.floor(self.rx * img.shape[-1]))
                y = int(np.floor(self.ry * img.shape[-2]))
                # 1st quadrant.
                if self.quad[0]:
                    self.perturb[0](img[...,:y,:x])
                # 2nd quadrant.
                if self.quad[1]:
                    self.perturb[1](img[...,y:,:x])
                # 3nd quadrant.
                if self.quad[2]:
                    self.perturb[2](img[...,:y,x:])
                # 4nd quadrant.
                if self.quad[3]:
                    self.perturb[3](img[...,y:,x:])

        rx, ry = np.random.rand(2)
        quad = np.random.rand(4) > 0.5
        perturb = [self.perturb(**self.params) for _ in range(4)]
        return _PerturbQuadrant(perturb, rx, ry, quad)


class MixedSection(PartialSection):
    def get_perturb(self):
        if np.random.rand() > 0.5:
            return Section.get_perturb(self)
        else:
            return PartialSection.get_perturb(self)
