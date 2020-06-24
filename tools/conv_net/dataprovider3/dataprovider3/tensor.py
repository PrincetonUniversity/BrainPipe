import math
import numpy as np
import time

from .geometry import Box, Vec3d, centered_box
from . import utils


class TensorData(object):
    """Read-only tensor data.

    The 1st dimension is regarded as channels, and arbitrary access
    in this dimension is not allowed. Threfore, every data access should be
    made through a 3-tuple, not 4-tuple.

    Args:
        data (ndarray): volumetric data of size (z,y,x) or (c,z,y,x).
        offset (3-tuple of int, optional): offset from the origin.

    Attributes:
        _data (ndarray): 4D numpy array. (channel, z, y, x)
        _dim (Vec3d):    Dimension of each channel.
        _offset (Vec3d): Coordinate offset from the origin.
        _bbox (Box):     Bounding box.
    """
    def __init__(self, data, offset=(0,0,0)):
        self._data = utils.to_tensor(data)
        self._dim = Vec3d(self._data.shape[-3:])
        self._offset = Vec3d(offset)

        # Set a bounding box.
        self._bbox = Box((0,0,0), self._dim)
        self._bbox.translate(self._offset)

    def get_patch(self, pos, dim):
        """Extract a patch of size `dim` centered on `pos`."""
        assert(len(pos)==3 and len(dim)==3)
        patch = None

        # Is the patch contained within the bounding box?
        box = centered_box(pos, dim)
        if self._bbox.contains(box):
            box.translate(-self._offset)  # Local coordinate system
            vmin = box.min()
            vmax = box.max()
            patch = np.copy(self._data[...,vmin[0]:vmax[0],
                                           vmin[1]:vmax[1],
                                           vmin[2]:vmax[2]])
        return patch

    def valid_range(self, dim):
        """Get a valid range for extracting patches of size `dim`."""
        assert(len(dim)==3)
        if any([dim[i] > self._dim[i] for i in range(3)]):
            return None
        dim = Vec3d(dim)
        top = dim // 2             # Top margin
        btm = dim - top - (1,1,1)  # Bottom margin
        vmin = self._offset + top
        vmax = self._offset + self._dim - btm
        return Box(vmin, vmax)

    ####################################################################
    ## Public methods for accessing attributes.
    ####################################################################

    def data(self):
        return self._data

    def shape(self):
        """Return data shape (c,z,y,x)."""
        return self._data.shape

    def dim(self):
        """Return data dim (z,y,x)."""
        return tuple(self._dim)

    def offset(self):
        return Vec3d(self._offset)

    def bbox(self):
        return Box(self._bbox)

    ####################################################################
    ## Private helper methods.
    ####################################################################

    def __str__( self ):
        return "<TensorData>\nshape: %s\ndim: %s\noffset: %s\n" % \
               (self.shape(), self.dim(), self.offset())


class WritableTensorData(TensorData):
    """
    Writable tensor data.
    """
    def __init__(self, data, offset=(0,0,0), fov=(0,0,0)):
        if isinstance(data, np.ndarray):
            super(WritableTensorData, self).__init__(data, offset)
        else:
            assert(len(data) >= 3)
            data = np.full(data, 0, dtype='float32')
            super(WritableTensorData, self).__init__(data, offset)

    def set_patch(self, pos, patch, op=None):
        assert(len(pos) == 3)
        patch = utils.to_tensor(patch)
        dim = patch.shape[-3:]
        box = centered_box(pos, dim)
        assert(self._bbox.contains(box))

        box.translate(-self._offset)  # Local coordinate system
        vmin = box.min()
        vmax = box.max()
        slices = [slice(None)] + [slice(vmin[i],vmax[i]) for i in range(3)]
        if op:
            self._data[tuple(slices)] = op(self._data[slices], patch)
        else:
            self._data[tuple(slices)] = patch


class WritableTensorDataWithMask(WritableTensorData):
    """
    Writable tensor data with blending mask.
    """
    def __init__(self, data, offset=(0,0,0)):
        super(WritableTensorDataWithMask, self).__init__(data, offset)
        self._norm = WritableTensorData(self.dim(), offset)
        self.normalized = False

    def set_patch(self, pos, patch, op=np.add, mask=None):
        # Default mask
        if mask is None:
            mask = np.full(patch.shape[-3:], 1, dtype='float32')
        else:
            mask = utils.to_volume(mask)
        t0 = time.time()
        WritableTensorData.set_patch(self, pos, patch*mask, op=op)
        t1 = time.time()
        self._norm.set_patch(pos, mask, op=np.add)
        t2 = time.time()
        # print("set_patch: %.3f, set_mask: %.3f" % (t1 - t0, t2 - t1))

    def norm(self):
        return self._norm._data

    def data(self):
        if not self.normalized:
            np.divide(self._data, self._norm._data, out=self._data)
            self.normalized = True
        return self._data

    def unnormalized_data(self):
        return np.multiply(self.data(), self._norm._data)


########################################################################
## Unit Testing
########################################################################
if __name__ == "__main__":

    import unittest

    ####################################################################
    class UnitTestTensorData(unittest.TestCase):

        def setup(self):
            pass

        def testCreation(self):
            data = np.zeros((4,4,4,4))
            T = TensorData(data, (1,1,1))
            self.assertTrue(T.shape()==(4,4,4,4))
            self.assertTrue(T.offset()==(1,1,1))
            bb = T.bbox()
            self.assertTrue(bb==Box((1,1,1),(5,5,5)))

        def testGetPatch(self):
            # (4,4,4) random 3D araray.
            data = np.random.rand(4,4,4)
            dim = (3,3,3)
            T = TensorData(data)
            p = T.get_patch((2,2,2), dim)
            self.assertTrue(np.array_equal(data[1:,1:,1:], p[0,...]))
            dim = (2,2,2)
            p = T.get_patch((2,2,2), dim)
            self.assertTrue(np.array_equal(data[1:3,1:3,1:3], p[0,...]))
            p = T.get_patch((3,3,3), (3,3,3))
            self.assertEqual(p, None)

    ####################################################################
    unittest.main()

    ####################################################################
