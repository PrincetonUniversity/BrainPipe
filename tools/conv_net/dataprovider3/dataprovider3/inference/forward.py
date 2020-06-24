import math
import numpy as np

from . import blend
from ..geometry import *


class ForwardScanner(object):
    """
    Forward scanner.
    """
    def __init__(self, dataset, scan_spec, **params):
        self._init()
        self.dataset = dataset
        self.scan_spec = dict(scan_spec)
        self.params = params
        self._setup()
        print(params)
    
    def pull(self, verbose=True):
        ret = None
        if self.counter < len(self.locs):
            assert(self.current is None)
            idx = self.counter
            loc = self.locs[idx]
            if verbose: print("({}/{}) loc: {}".format(idx+1, len(self.locs), tuple(loc))) #zmd added
            ret = self.dataset.get_sample(loc)
            self.current = loc
            self.counter += 1
        return ret

    def push(self, sample):
        assert(self.current is not None)
        self.outputs.push(self.current, sample)
        self.current = None

    def voxels(self):
        return self.outputs.voxels()

    ####################################################################
    ## Private Methods.
    ####################################################################

    def _init(self):
        """Initialize all attributes."""
        self.dataset        = None
        self.scan_spec      = dict()
        self.params         = dict()
        self.offset         = (0,0,0)
        self.stride         = (0,0,0)
        self.grid           = (0,0,0)
        self.vmin           = None
        self.vmax           = None
        self.default_stride = None
        self.coords         = [None]*3
        self.locs           = list()
        self.counter        = 0
        self.current        = None
        self.outputs        = None

    def _setup(self):
        self.offset = Vec3d(self.params.get('offset', (0,0,0)))
        self.stride = Vec3d(self.params.get('stride', (0,0,0)))
        self.grid   = Vec3d(self.params.get('grid',   (0,0,0)))
                
        self.vmin = self.dataset.valid_range().min() + self.offset
        self.vmax = self.dataset.valid_range().max()

        # Order is important!
        self._setup_stride()
        self._setup_coords()
        self._prepare_outputs()

    def _setup_stride(self):
        stride = None
        for k, v in self.scan_spec.items():
            box = centered_box(Vec3d(0,0,0), v[-3:])
            if stride is None:
                stride = box
            else:
                stride = stride.intersect(box)
        assert(stride is not None)
        self.default_stride = stride.size()

    def _setup_coords(self):
        self._setup_coord(0)  # z-dimension
        self._setup_coord(1)  # y-dimension
        self._setup_coord(2)  # x-dimension
        self.locs = list()
        for z in self.coords[0]:
            for y in self.coords[1]:
                for x in self.coords[2]:
                    self.locs.append(Vec3d(z,y,x))

    def _setup_coord(self, dim):
        assert(dim < 3)

        # Min & max coordinates
        cmin = int(self.vmin[dim])
        cmax = int(self.vmax[dim])
        assert(cmin < cmax)

        # Dimension-specific params
        stride = self.stride[dim]
        grid = int(self.grid[dim])
        coord = set()

        # Stride
        if stride == 0:
            # Non-overlapping stride
            stride = self.default_stride[dim]
        elif stride > 0 and stride < 1:
            # Overlapping stride given by an overlapping ratio
            stride = math.ceil(stride * self.default_stride[dim])
        stride = int(stride)
        self.stride[dim] = stride

        # Automatic full spanning
        if grid == 0:
            grid = int((cmax - cmin - 1)//stride + 1)
            coord.add(cmax - 1)  # Offcut

        # Scan coords
        for i in range(grid):
            c = cmin + i*stride
            if c >= cmax:
                break
            coord.add(c)

        # Sanity check
        assert((cmin + (grid - 1)*stride) < cmax)

        # Sort coords
        self.coords[dim] = sorted(coord)

    def _prepare_outputs(self):
        """Prepare outputs according to the blending mode."""
        # Inference with overlapping windows
        diff = self.stride - self.default_stride
        overlap = any([x < 0 for x in diff])

        # Prepare outputs.
        blend_mode = self.params.get('blend', '')
        self.outputs = blend.prepare_outputs(
            self.scan_spec, self.locs,
            blend=overlap, blend_mode=blend_mode
        )
