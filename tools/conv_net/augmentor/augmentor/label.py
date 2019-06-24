from __future__ import print_function
import numpy as np

try:
    import datatools
    dtools = True
except ImportError:
    import skimage.measure as measure
    dtools = False

from .augment import Augment, Compose


class Label(Augment):
    """
    Recompute connected components.
    """
    def __init__(self):
        self.segs = []

    def prepare(self, spec, segs=[], **kwargs):
        self.segs = self._validate(spec, segs)
        return dict(spec)

    def __call__(self, sample, **kwargs):
        sample = Augment.to_tensor(sample)
        for k in self.segs:
            seg = sample[k][0,:,:,:].astype('uint32')
            if dtools:
                aff = datatools.make_affinity(seg)
                sample[k] = datatools.get_segmentation(aff).astype('uint32')
            else:
                sample[k] = measure.label(seg).astype('uint32')
        return Augment.sort(Augment.to_tensor(sample))

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string

    def _validate(self, spec, segs):
        assert len(segs) > 0
        assert all(k in spec for k in segs)
        return segs
