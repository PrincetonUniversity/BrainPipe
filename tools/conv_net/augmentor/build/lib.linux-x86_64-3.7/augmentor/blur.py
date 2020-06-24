from __future__ import print_function
import collections

from .augment import Augment
from .section import Section, PartialSection, MixedSection
from . import perturb


__all__ = ['BlurrySection', 'PartialBlurrySection', 'MixedBlurrySection']


class BlurrySection(Section):
    """
    Simulate full out-of-focus sections in a training sample.
    """
    def __init__(self, sigma=5.0, random=True, **kwargs):
        super(BlurrySection, self).__init__(perturb.Blur, **kwargs)
        self.params = dict(sigma=sigma, random=random)


class PartialBlurrySection(PartialSection):
    """
    Simulate partial out-of-focus sections in a training sample.
    """
    def __init__(self, sigma=5.0, random=True, **kwargs):
        super(PartialBlurrySection, self).__init__(perturb.Blur, **kwargs)
        self.params = dict(sigma=sigma, random=random)


class MixedBlurrySection(MixedSection):
    """
    Simulate full & partial out-of-focus sections.
    """
    def __init__(self, sigma=5.0, random=True, **kwargs):
        super(MixedBlurrySection, self).__init__(perturb.Blur, **kwargs)
        self.params = dict(sigma=sigma, random=random)
