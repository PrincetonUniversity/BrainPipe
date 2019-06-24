from collections import OrderedDict
import copy
import numpy as np

from .geometry import Box, Vec3d
from .tensor import TensorData
from . import utils


class Dataset(object):
    """Dataset for volumetric data.

    Attributes:
        spec (dict): mapping key to tensor's shape.
        data (dict): mapping key to TensorData.
        locs (dict): valid locations.
    """
    def __init__(self, spec=None, tag=''):
        self.set_spec(spec)
        self.tag = tag
        self.data = dict()
        self.locs = None

    def __call__(self, spec=None):
        return self.random_sample(spec=spec)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += self.tag
        format_string += ')'
        return format_string

    def add_data(self, key, data, offset=(0,0,0)):
        self.data[key] = TensorData(data, offset=offset)

    def add_mask(self, key, data, offset=(0,0,0), loc=False):
        self.add_data(key, data, offset=offset)
        if loc:
            self.locs = dict()
            self.locs['data'] = np.flatnonzero(data)
            self.locs['dims'] = data.shape
            self.locs['offset'] = Vec3d(offset)

    def set_spec(self, spec):
        self.spec = None
        if spec is not None:
            self.spec = dict(spec)

    def get_patch(self, key, pos, dim):
        """Extract a patch from the data tagged with `key`."""
        assert key in self.data
        assert len(pos)==3 and len(dim)==3
        return self.data[key].get_patch(pos, dim)

    def get_sample(self, pos, spec=None):
        """Extract a sample centered on pos."""
        spec = self._validate(spec)
        sample = dict()
        for key, dim in spec.items():
            patch = self.get_patch(key, pos, dim[-3:])
            if patch is None:
                raise Dataset.OutOfRangeError()
            sample[key] = patch
        return utils.sort(sample)

    def random_sample(self, spec=None):
        """Extract a random sample."""
        spec = self._validate(spec)
        try:
            pos = self._random_location(spec)
            ret = self.get_sample(pos, spec)
        except Dataset.OutOfRangeError:
            print("out-of-range error")
            raise
        except:
            raise
        return ret

    def num_samples(self, spec=None):
        try:
            if self.locs is None:
                spec = self._validate(spec)
                valid = self._valid_range(spec)
                num = np.prod(valid.size())
            else:
                num = self.locs['data'].size
        except Dataset.NoSpecError:
            nums = list()
            for k, v in self.data.items():
                nums.append(np.prod(v.dim()))
            num = min(nums)
        except:
            raise
        return num

    def valid_range(self, spec=None):
        spec = self._validate(spec)
        return self._valid_range(spec)

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _validate(self, spec):
        if spec is None:
            if self.spec is None:
                raise Dataset.NoSpecError()
            spec = dict(self.spec)
        assert all([k in self.data for k in spec])
        return spec

    def _random_location(self, spec):
        """Return a random valid location."""
        valid = self._valid_range(spec)
        if self.locs is None:
            s = tuple(valid.size())
            x = np.random.randint(0, s[-1])
            y = np.random.randint(0, s[-2])
            z = np.random.randint(0, s[-3])
            # Global coordinate system.
            loc = Vec3d(z,y,x) + valid.min()
        else:
            while True:
                idx = np.random.choice(self.locs['data'], 1)
                loc = np.unravel_index(idx[0], self.locs['dims'])
                # Global coordinate system.
                loc = Vec3d(loc[-3:]) + self.locs['offset']
                if valid.contains(loc):
                    break
        # DEBUG:
        # print('loc = {}'.format(loc))
        return loc

    def _valid_range(self, spec):
        """Compute the valid range, which is intersection of the valid range
        of each TensorData.
        """
        valid = None
        for key, dim in spec.items():
            assert key in self.data
            v = self.data[key].valid_range(dim[-3:])
            if v is None:
                raise Dataset.OutOfRangeError()
            valid = v if valid is None else valid.intersect(v)
        assert valid is not None
        return valid

    ####################################################################
    ## Exceptions.
    ####################################################################

    class OutOfRangeError(Exception):
        pass

    class NoSpecError(Exception):
        pass
