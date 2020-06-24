from __future__ import print_function
import numpy as np

from .augment import Augment, Blend
from .perturb import Grayscale
from .section import Section, PartialSection, MixedSection


__all__ = ['Grayscale2D', 'Grayscale3D', 'GrayscaleMixed',
           'PartialGrayscale2D', 'MixedGrayscale2D']


class Grayscale3D(Augment):
    """Grayscale value perturbation.

    Randomly adjust contrast/brightness, and apply random gamma correction.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, skip=0.3):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.skip = np.clip(skip, 0, 1)
        self.do_aug = False
        self.imgs = []

    def prepare(self, spec, imgs=[], **kwargs):
        # Biased coin toss.
        self.do_aug = np.random.rand() > self.skip
        self.imgs = self._validate(spec, imgs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            perturb = Grayscale(self.contrast_factor,
                                self.brightness_factor)
            for k in self.imgs:
                perturb(sample[k])
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast_factor={0}, '.format(self.contrast_factor)
        format_string += 'brightness_factor={0}, '.format(self.brightness_factor)
        format_string += 'skip={:.2f}'.format(self.skip)
        format_string += ')'
        return format_string

    def _validate(self, spec, imgs):
        assert len(imgs) > 0
        assert all(k in spec for k in imgs)
        return imgs


class Grayscale2D(Section):
    """
    Perturb each z-slice independently.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, prob=1,
                 **kwargs):
        super(Grayscale2D, self).__init__(Grayscale, prob=prob, **kwargs)
        self.params = dict(contrast_factor=contrast_factor,
                           brightness_factor=brightness_factor)


class GrayscaleMixed(Blend):
    """
    Half 2D & half 3D.
    """
    def __init__(self, **kwargs):
        grayscales = [Grayscale2D(**kwargs), Grayscale3D(**kwargs)]
        super(GrayscaleMixed, self).__init__(grayscales)


class PartialGrayscale2D(PartialSection):
    """
    Perturb each z-slice independently.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, **kwargs):
        super(PartialGrayscale2D, self).__init__(Grayscale, **kwargs)
        self.params = dict(contrast_factor=contrast_factor,
                           brightness_factor=brightness_factor)


class MixedGrayscale2D(MixedSection):
    """
    Perturb each z-slice independently.
    """
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, **kwargs):
        super(MixedGrayscale2D, self).__init__(Grayscale, **kwargs)
        self.params = dict(contrast_factor=contrast_factor,
                           brightness_factor=brightness_factor)
