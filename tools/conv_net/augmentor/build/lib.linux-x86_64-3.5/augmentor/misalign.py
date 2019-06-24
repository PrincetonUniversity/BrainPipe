from __future__ import print_function
import numpy as np

from .augment import Augment, Blend
from .flip import FlipRotate
from .track import Track
from . import utils


__all__ = ['Misalign','MisalignPlusMissing',
           'MisalignTrackMissing','SlipMisalign']


class Misalign(Augment):
    """Translational misalignment.

    Args:
        disp (2-tuple of int): Min/max displacement.
        margin (int):

    TODO:
        1. Valid architecture
        2. Augmentation territory
    """
    def __init__(self, disp, margin=0):
        self.disp = disp
        self.margin = max(margin, 0)
        self.tx = 0
        self.ty = 0
        self.zmin = 2
        self.flip_rotate = FlipRotate()

    def prepare(self, spec, **kwargs):
        spec = self.flip_rotate.prepare(spec, **kwargs)

        # Original spec
        self.spec = dict(spec)

        # Random displacement in x/y dimension.
        self.tx = np.random.randint(*self.disp)
        self.ty = np.random.randint(*self.disp)

        # Increase tensor dimension by the amount of displacement.
        zdims = dict()
        spec = dict(spec)
        for k, shape in spec.items():
            z, y, x = shape[-3:]
            zdims[k] = z
            spec[k] = shape[:-2] + (y + self.ty, x + self.tx)

        # Pick a section to misalign.
        zmin = min(zdims.values())
        assert zmin >= 2*self.margin + self.zmin
        zloc = np.random.randint(self.margin + 1, zmin - self.margin)

        # Offset z-location.
        self.zlocs = dict()
        for k, zdim in zdims.items():
            offset = (zdim - zmin) // 2
            self.zlocs[k] = offset + zloc

        return spec

    def __call__(self, sample, **kwargs):
        return self.flip_rotate(self.misalign(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'disp={0}, '.format(self.disp)
        format_string += 'margin={0}'.format(self.margin)
        format_string += ')'
        return format_string

    def misalign(self, sample):
        sample = Augment.to_tensor(sample)

        for k, v in sample.items():
            # New tensor
            w = np.zeros(self.spec[k], dtype=v.dtype)
            w = utils.to_tensor(w)

            # Misalign.
            z, y, x = w.shape[-3:]
            zloc = self.zlocs[k]
            w[:,:zloc,...] = v[:,:zloc,:y,:x]
            w[:,zloc:,...] = v[:,zloc:,-y:,-x:]
            sample[k] = w

        return Augment.sort(sample)


class MisalignPlusMissing(Misalign):
    """
    Translational misalignment + missing section(s).
    """
    def __init__(self, disp, margin=1, value=0, random=False):
        margin = max(margin, 1)
        super(MisalignPlusMissing, self).__init__(disp, margin=margin)
        assert self.margin > 0
        self.value = value
        self.random = random
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(MisalignPlusMissing, self).prepare(spec, **kwargs)
        self.both = np.random.rand() > 0.5
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        sample = self.misalign(sample)
        sample = self.missing(sample)
        sample = self.flip_rotate(sample)
        return Augment.sort(sample)

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs

    def misalign(self, sample):
        for k, v in sample.items():
            # New tensor
            w = np.zeros(self.spec[k], dtype=v.dtype)
            w = utils.to_tensor(w)

            # Misalign.
            z, y, x = w.shape[-3:]
            zloc = self.zlocs[k]
            w[:,:zloc,...] = v[:,:zloc,:y,:x]
            w[:,zloc:,...] = v[:,zloc:,-y:,-x:]

            if k not in self.imgs:
                # Target interpolation
                if self.both:
                    tx = round(self.tx / 3.0)
                    ty = round(self.ty / 3.0)
                    w[:,zloc-1,...] = v[:,zloc-1,ty:ty+y,tx:tx+x]
                    w[:,zloc,...] = v[:,zloc,-ty-y:-ty,-tx-x:-tx]
                else:
                    tx = round(self.tx / 2.0)
                    ty = round(self.ty / 2.0)
                    w[:,zloc,...] = v[:,zloc,ty:ty+y,tx:tx+x]

            # Update sample.
            sample[k] = w

        return sample

    def missing(self, sample):
        val = np.random.rand() if self.random else self.value

        for k in self.imgs:
            zloc = self.zlocs[k]
            img = sample[k]
            img[:,zloc,...] = val
            if self.both:
                img[:,zloc-1,...] = val
            sample[k] = img

        return sample


class MisalignTrackMissing(MisalignPlusMissing):
    """
    Translational misalignment + track mark + missing section(s).
    """
    def __init__(self, track, disp, **kwargs):
        assert isinstance(track, Track)
        super(MisalignTrackMissing, self).__init__(disp, **kwargs)
        self.track = track
        self.track.flip_rotate = None

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(MisalignTrackMissing, self).prepare(spec, imgs=imgs, **kwargs)
        spec = self.track.prepare(spec, imgs=imgs, **kwargs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        sample = self.misalign(sample)
        sample = self.track(sample)
        sample = self.missing(sample)
        sample = self.flip_rotate(sample)
        return Augment.sort(sample)


class SlipMisalign(Misalign):
    def __init__(self, disp, margin=1, interp=False):
        margin = max(margin, 1)
        super(SlipMisalign, self).__init__(disp, margin=margin)
        self.zmin = 1
        self.interp = interp
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        spec = super(SlipMisalign, self).prepare(spec, **kwargs)
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        return self.flip_rotate(self.misalign(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'disp={0}, '.format(self.disp)
        format_string += 'margin={0}, '.format(self.margin)
        format_string += 'interp={0}'.format(self.interp)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs

    def misalign(self, sample):
        sample = Augment.to_tensor(sample)

        for k, v in sample.items():
            # New tensor
            w = np.zeros(self.spec[k], dtype=v.dtype)
            w = utils.to_tensor(w)

            # Misalign.
            z, y, x = w.shape[-3:]
            zloc = self.zlocs[k]
            w[...] = v[...,:y,:x]
            if (k in self.imgs) or (not self.interp):
                w[:,zloc,...] = v[:,zloc,-y:,-x:]
            sample[k] = w

        return Augment.sort(sample)
