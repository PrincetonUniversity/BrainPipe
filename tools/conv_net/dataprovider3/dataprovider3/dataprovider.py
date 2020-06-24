import numpy as np

from .dataset import Dataset


class DataProvider(object):
    """DataProvider.

    Args:
        spec (dict):
        dsets (list of Dataset): Datasets.
        augs (Augment): Augment.
        p (list of float): sampling weights.
    """
    def __init__(self, spec):
        self.spec = dict(spec)
        self.datasets = list()
        self.augments = None
        self.p = None

    def add_dataset(self, dset):
        assert isinstance(dset, Dataset)
        assert all([k in dset.data for k in self.spec])
        self.datasets.append(dset)

    def set_augment(self, aug):
        self.augments = aug

    def set_imgs(self, imgs):
        assert len(imgs) > 0
        self.imgs = list(imgs)

    def set_segs(self, segs):
        assert len(segs) > 0
        self.segs = list(segs)

    def set_sampling_weights(self, p=None):
        if p is None:
            p = [d.num_samples() for d in self.datasets]
        p = np.asarray(p, dtype='float32')
        p = p/np.sum(p)
        assert len(p)==len(self.datasets)
        self.p = p

    def random_dataset(self):
        assert len(self.datasets) > 0
        if self.p is None:
            self.set_sampling_weights()
        idx = np.random.choice(len(self.datasets), size=1, p=self.p)
        return self.datasets[idx[0]]

    def random_sample(self):
        dset = self.random_dataset()
        while True:
            try:
                spec = dict(self.spec)
                imgs = list(self.imgs)
                segs = list(self.segs)
                if self.augments is None:
                    sample = dset(spec=spec)
                else:
                    spec = self.augments.prepare(spec, imgs=imgs, segs=segs)
                    sample = self.augments(dset(spec=spec))
                break
            except Dataset.OutOfRangeError:
                pass
            except:
                raise
        return sample

    def __call__(self):
        return self.random_sample()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for i, k in enumerate(self.datasets):
            format_string += '\n'
            format_string += '    {0:.3f} : {1}'.format(self.p[i], k)
        format_string += '\n)'
        return format_string
