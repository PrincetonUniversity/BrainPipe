from __future__ import print_function
import collections

from .augment import Augment
from .section import Section, PartialSection, MixedSection
from . import perturb


__all__ = ['MissingSection', 'PartialMissingSection', 'MixedMissingSection']


class MissingSection(Section):
    """
    Simulate full missing sections in a training sample.
    """
    def __init__(self, value=0, random=True, **kwargs):
        super(MissingSection, self).__init__(perturb.Fill, **kwargs)
        self.params = dict(value=value, random=random)


class PartialMissingSection(PartialSection):
    """
    Simulate partial missing sections in a training sample.
    """
    def __init__(self, value=0, random=True, **kwargs):
        super(PartialMissingSection, self).__init__(perturb.Fill, **kwargs)
        self.params = dict(value=value, random=random)


class MixedMissingSection(MixedSection):
    """
    Mixed full & partial missing sections.
    """
    def __init__(self, value=0, random=True, **kwargs):
        super(MixedMissingSection, self).__init__(perturb.Fill, **kwargs)
        self.params = dict(value=value, random=random)
