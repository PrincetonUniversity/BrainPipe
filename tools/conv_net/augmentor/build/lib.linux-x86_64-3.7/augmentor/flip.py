from __future__ import print_function
import numpy as np

from .augment import Augment, Compose


__all__ = ['Flip', 'Transpose', 'FlipRotate']


class Flip(Augment):
    """Random flip.

    Args:
        axis (int):
        prob (float, optional):
    """
    def __init__(self, axis, prob=0.5):
        self.axis = axis
        self.prob = np.clip(prob, 0, 1)
        self.do_aug = False

    def prepare(self, spec, **kwargs):
        # Biased coin toss
        self.do_aug = np.random.rand() < self.prob
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            for k, v in sample.items():
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.flip(v, self.axis))
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'axis={}, '.format(self.axis)
        format_string += 'prob={:.3f}'.format(self.prob)
        format_string += ')'
        return format_string


class Transpose(Augment):
    """Random transpose.

    Args:
        axes (list of int, optional):
        prob (float, optional):
    """
    def __init__(self, axes=None, prob=0.5):
        assert (axes is None) or (len(axes)==4)
        self.axes = axes
        self.prob = np.clip(prob, 0, 1)
        self.do_aug = False

    def prepare(self, spec, **kwargs):
        spec = dict(spec)
        # Biased coin toss
        self.do_aug = np.random.rand() < self.prob
        if (not self.do_aug) or (self.axes is None):
            return spec
        for k, v in spec.items():
            assert len(v)==3 or len(v)==4
            offset = 1 if len(v)==3 else 0
            spec[k] = tuple(v[:-3]) + tuple(v[x - offset] for x in self.axes[-3:])
        return spec

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        if self.do_aug:
            for k, v in sample.items():
                # Prevent potential negative stride issues by copying.
                sample[k] = np.copy(np.transpose(v, self.axes))
        return Augment.sort(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'axes={}, '.format(self.axes)
        format_string += 'prob={:.3f}'.format(self.prob)
        format_string += ')'
        return format_string


class FlipRotate(Compose):
    def __init__(self):
        augs = [
            Flip(axis=-1),
            Flip(axis=-2),
            Flip(axis=-3),
            Transpose(axes=[0,1,3,2])
        ]
        super(FlipRotate, self).__init__(augs)
