from __future__ import print_function
from collections import OrderedDict
import numpy as np

from . import utils


class Augment(object):
    """
    Abstract interface.
    """
    def __init__(self):
        raise NotImplementedError

    def prepare(self, spec, **kwargs):
        return dict(spec)

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    @staticmethod
    def to_tensor(sample):
        """Ensure that every data in sample is a tensor."""
        for k, v in sample.items():
            sample[k] = utils.to_tensor(v)
        return sample

    @staticmethod
    def sort(sample):
        """Ensure that sample is sorted by key."""
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))

    @staticmethod
    def get_spec(sample):
        """Extract spec from sample."""
        spec = dict()
        for k, v in sample.items():
            spec[k] = v.shape[-3:]
        return spec


class Compose(Augment):
    """Composes several augments together.

    Args:
        augments (list of ``Augment`` objects): list of augments to compose.
    """
    def __init__(self, augments):
        self.augments = augments

    def prepare(self, spec, **kwargs):
        for aug in reversed(self.augments):
            spec = aug.prepare(spec, **kwargs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        for aug in self.augments:
            sample = aug(sample, **kwargs)
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for aug in self.augments:
            format_string += '\n'
            format_string += '    {}'.format(aug)
        format_string += '\n)'
        return format_string


class Blend(Augment):
    """Blends several augments together.

    Args:
        augments (list of ``Augment`` objects): List of augments to blend.
        props (list of floats, optional): Blending proportions.
    """
    def __init__(self, augments, props=None):
        self.augments = augments
        self.props = [1.0]*len(augments) if props is None else props
        self.props /= np.sum(self.props)  # Normalize.
        assert len(self.augments)==len(self.props)
        self.aug = None

    def prepare(self, spec, **kwargs):
        """Choose an augment and prepare it."""
        self.aug = self._choose()
        return self.aug.prepare(spec, **kwargs)

    def __call__(self, sample, **kwargs):
        # Lazy prep.
        if self.aug is None:
            spec = Augment.get_spec(sample)
            self.prepare(spec)
        return self.aug(sample, **kwargs)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for prop, aug in zip(self.props, self.augments):
            format_string += '\n'
            format_string += '    {0:.3f} : {1}'.format(prop, aug)
        format_string += '\n)'
        return format_string

    def _choose(self):
        idx = np.random.choice(len(self.props), size=1, p=self.props)
        return self.augments[idx[0]]
