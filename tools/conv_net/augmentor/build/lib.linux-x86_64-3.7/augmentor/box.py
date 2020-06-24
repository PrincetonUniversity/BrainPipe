from __future__ import print_function
import numpy as np

from .augment import Augment
from .geometry.box import Box, centered_box
from .geometry.vector import Vec3d
from .perturb import Perturb


__all__ = ['FillBox', 'BlurBox', 'NoiseBox']


class BoxOcclusion(Augment):
    """
    Box occlusion.

    Args:
        perturb_cls (``Perturb``): ``Perturb`` class.
        dims (2-tpule): min/max dimension of box.
        aniso (int): anisotropy factor.
        density (float):
        margin (3-tuple, optional):
        skip (float, optional): skip probability.
    """
    def __init__(self, perturb_cls, dims=(10,30), aniso=10, density=0.5,
                 margin=(0,0,0), individual=True, skip=0, **params):
        assert issubclass(perturb_cls, Perturb)
        self.perturb_cls = perturb_cls
        self.dims = dims
        self.aniso = aniso
        self.density = np.clip(density, 0, 1)
        self.margin = margin
        self.individual = individual
        self.skip = np.clip(skip, 0, 1)
        self.params = params
        self.do_aug = False
        self.perturb = None

    def prepare(self, spec, imgs=[], **kwargs):
        self.do_aug = np.random.rand() > self.skip
        self.perturb = self.get_perturb() if self.do_aug else None
        self.spec = dict(spec)
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            sample = self.augment(sample, **kwargs)
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        # TODO(kisuk): add properties.
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs

    def get_perturb(self):
        return self.perturb_cls(**self.params)

    def augment(self, sample, **kwargs):
        # Find union of bounding boxes.
        bbox_union = None
        self.bbox = dict()
        for k in self.imgs:
            dim = self.spec[k][-3:]
            box = centered_box((0,0,0), dim)
            bbox_union = box if bbox_union is None else bbox_union.merge(box)
            self.bbox[k] = box

        # Create a mask.
        offset = bbox_union.min()
        bbox_dim = bbox_union.size()

        # Random box density
        count = 0
        density = self.density * np.random.rand()
        goal = bbox_union.volume() * density

        while True:
            # Random location
            loc = [0,0,0]
            for i in range(3):
                m = self.margin[i]
                loc[i] = np.random.randint(m, bbox_dim[i] - m)
            loc = Vec3d(loc) + offset

            # Random box size
            dim = np.random.randint(self.dims[0], self.dims[1] + 1, 3)
            dim[0] //= int(self.aniso)
            dim[0] = max(dim[0], 1)

            # Random box
            box = centered_box(loc, dim)

            # Perturb each individual box independently.
            if self.individual:
                self.perturb = self.get_perturb()

            for k in self.imgs:
                bbox = self.bbox[k]
                box2 = bbox.intersect(box)
                if box2 is None:
                    continue
                box2.translate(-offset)
                vmin = box2.min()
                vmax = box2.max()
                s0 = slice(vmin[0],vmax[0])
                s1 = slice(vmin[1],vmax[1])
                s2 = slice(vmin[2],vmax[2])
                assert self.perturb is not None
                self.perturb(sample[k][...,s0,s1,s2])

            # Stop condition
            count += box.volume()
            if count > goal:
                break

        for k in self.imgs:
            sample[k] = np.clip(sample[k], 0, 1)

        return sample


from .perturb import Fill, Blur, Noise


class FillBox(BoxOcclusion):
    def __init__(self, value=0, random=True, **kwargs):
        super(FillBox, self).__init__(Fill, **kwargs)
        self.params = dict(value=value, random=random)


class BlurBox(BoxOcclusion):
    def __init__(self, sigma=5.0, random=True, **kwargs):
        super(BlurBox, self).__init__(Blur, **kwargs)
        self.params = dict(sigma=sigma, random=random)


class NoiseBox(BoxOcclusion):
    def __init__(self, sigma=(2,5), **kwargs):
        super(NoiseBox, self).__init__(Noise, **kwargs)
        self.params = dict(sigma=sigma)
