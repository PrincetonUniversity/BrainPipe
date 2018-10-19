from neurotorch.datasets.dataset import Volume, Array, Data
from neurotorch.datasets.datatypes import BoundingBox, Vector
from abc import abstractmethod
import fnmatch
import os.path
import h5py
import numpy as np
import tifffile as tif


class TiffVolume(Volume):
    def __init__(self, tiff_file, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Loads a TIFF stack file or a directory of TIFF files and creates a
corresponding three-dimensional volume dataset
        :param tiff_file: Either a TIFF stack file or a directory
containing TIFF files
        :param chunk_size: Dimensions of the sample subvolume
        """
        # Set TIFF file and bounding box
        self.setFile(tiff_file)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, tiff_file):
        if os.path.isfile(tiff_file) or os.path.isdir(tiff_file):
            self.tiff_file = tiff_file
        else:
            raise IOError("{} was not found".format(tiff_file))

    def getFile(self):
        return self.tiff_file

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            try:
                array = tif.imread(self.getFile())

            except IOError:
                raise IOError("TIFF file {} could not be " +
                              "opened".format(self.getFile()))

        elif os.path.isdir(self.getFile()):
            tiff_list = os.listdir(self.getFile())
            tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                               tiff_list)

            #ben added
            tiff_list = [os.path.join(self.getFile(), f) for f in tiff_list]

            if tiff_list:
                array = np.squeeze(tif.TiffSequence(tiff_list).asarray()) #zmd modified

        else:
            raise IOError("{} was not found".format(self.getFile()))

        array = Array(array, bounding_box=self.getBoundingBox(),
                      iteration_size=self.getIterationSize(),
                      stride=self.getStride())
        self.setArray(array)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)


class LargeVolume(Volume):

    def __init__(self, bounding_box,
                 iteration_size=BoundingBox(Vector(0, 0, 0),
                                            Vector(128, 128, 20)),
                 stride=Vector(64, 64, 10)):
        self.setIteration(iteration_size=iteration_size,
                          stride=stride)
        super().__init__(bounding_box)

    @abstractmethod
    def get(self, bounding_box):
        pass

    def set(self, *args):
        raise RuntimeError("a LargeVolume is read-only")

    @abstractmethod
    def getBoundingBox(self):
        pass

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        if not isinstance(iteration_size, BoundingBox):
            error_string = ("iteration_size must have type BoundingBox"
                            + " instead it has type {}")
            error_string = error_string.format(type(iteration_size))
            raise ValueError(error_string)

        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")

        if not iteration_size.isSubset(self.getBoundingBox()):
            raise ValueError("iteration_size must be smaller than volume size")

        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return int(round(x))

        self.element_vec = Vector(*map(lambda L, l, s: ceil((L-l)/s+1),
                                       self.getBoundingBox().getEdges()[1].getComponents(),
                                       self.iteration_size.getEdges()[1].getComponents(),
                                       self.stride.getComponents()))

        self.index = 0

    def setIterationSize(self, iteration_size):
        self.iteration_size = BoundingBox(Vector(0, 0, 0),
                                          iteration_size.getSize())

    def setStride(self, stride):
        self.stride = stride

    def getIterationSize(self):
        return self.iteration_size

    def getStride(self):
        return self.stride

    def __len__(self):
        return self.element_vec[0]*self.element_vec[1]*self.element_vec[2]

    def __getitem__(self, idx):
        if idx >= len(self):
            self.index = 0
            raise StopIteration

        element_vec = np.unravel_index(idx,
                                       dims=self.element_vec.getComponents())

        element_vec = Vector(*element_vec)
        bounding_box = self.iteration_size+self.stride*element_vec
        result = self.get(bounding_box)

        return result


class LargeTiffVolume(LargeVolume):
    def __init__(self, tiff_dir, *args, **kwargs):
        self.setDirectory(tiff_dir)
        self.setCache()
        super().__init__(bounding_box=self.getBoundingBox())

    def get(self, bounding_box):
        if bounding_box.isDisjoint(self.getBoundingBox()):
            error_string = ("Bounding box must be inside dataset " +
                            "dimensions instead bounding box is {} while " +
                            "the dataset dimensions are {}")
            error_string = error_string.format(bounding_box,
                                               self.getBoundingBox())
            raise ValueError(error_string)

        sub_bounding_box = bounding_box.intersect(self.getBoundingBox())
        array = self.getArray(sub_bounding_box)

        before_pad = (bounding_box.getEdges()[0] -
                      sub_bounding_box.getEdges()[0])
        after_pad = sub_bounding_box.getEdges()[1] - bounding_box.getEdges()[1]

        if before_pad != Vector(0, 0, 0) and after_pad != Vector(0, 0, 0):
            pad_size = (before_pad.getComponents(),
                        after_pad.getComponents())
            array = np.pad(array, pad_size=pad_size, mode="constant")

        return Data(array, bounding_box)

    def setDirectory(self, tiff_dir):
        if not os.path.isdir(tiff_dir):
            raise ValueError("tiff_dir must be a valid directory")

        tiff_list = os.listdir(tiff_dir)
        tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                           tiff_list)
        tiff_list = list(map(lambda f: os.path.join(tiff_dir, f),
                             tiff_list))
        tiff_list.sort()

        self._setTiffList(tiff_list)

    def _setTiffList(self, tiff_list):
        self.tiff_list = tiff_list
        self.setShape()

    def getTiffList(self):
        return self.tiff_list

    def setShape(self):
        z = len(self.getTiffList())
        x, y = tif.imread(self.getTiffList()[0]).shape

        self.shape = (x, y, z)

    def getShape(self):
        return self.shape

    def getBoundingBox(self):
        return BoundingBox(Vector(0, 0, 0),
                           Vector(*self.getShape()))

    def getArray(self, bounding_box):
        if not bounding_box.isSubset(self.getBoundingBox()):
            raise ValueError("The bounding box must be a subset" +
                             " of the volume")

        if not bounding_box.isSubset(self.getCache().getBoundingBox()):
            edge1, edge2 = bounding_box.getEdges()
            x_len, y_len, z_len = self.getShape()
            cache_bbox = BoundingBox(Vector(0, 0, edge1[2]-50),
                                     Vector(x_len, y_len, edge2[2]+50))
            cache_bbox = cache_bbox.intersect(self.getBoundingBox())
            self.setCache(self, cache_bbox)

        return self.getCache().get(bounding_box).getArray()

    def setCache(self, bounding_box=None):
        if bounding_box is None:
            edge1, edge2 = self.getBoundingBox().getEdges()
            x1, y1, z1 = edge1
            x2, y2, z2 = edge2
            cache_bbox = BoundingBox(Vector(x1, y1, z1),
                                     Vector(x2, y2, z1+100))
            cache_bbox = cache_bbox.intersect(self.getBoundingBox())
            _bounding_box = cache_bbox
        else:
            _bounding_box = bounding_box

        if not _bounding_box.isSubset(self.getBoundingBox()):
            raise ValueError("cache bounding box must be a subset of " +
                             "volume bounding box")

        edge1, edge2 = _bounding_box.getEdges()
        x1, y1, z1 = edge1
        x2, y2, z2 = edge2

        array = [tif.imread(tiff_file)[y1:y2, x1:x2]
                 for tiff_file in self.getTiffList()[z1:z2]]
        array = list(map(lambda s: s.reshape(1, *s.shape),
                         array))
        array = np.concatenate(array)

        self.cache = Array(array)
        self.cache.setBoundingBox(_bounding_box)

    def getCache(self):
        return self.cache


class Hdf5Volume(Volume):
    def __init__(self, hdf5_file, dataset, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Loads a HDF5 dataset and creates a corresponding three-dimensional
volume dataset

        :param hdf5_file: A HDF5 file path
        :param dataset: A HDF5 dataset name
        :param chunk_size: Dimensions of the sample subvolume
        """
        self.setFile(hdf5_file)
        self.setDataset(dataset)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def getFile(self):
        return self.hdf5_file

    def setDataset(self, hdf5_dataset):
        self.hdf5_dataset = hdf5_dataset

    def getDataset(self):
        return self.hdf5_dataset

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            with h5py.File(self.getFile(), 'r') as f:
                array = f[self.getDataset()].value
                array = Array(array, bounding_box=self.getBoundingBox(),
                              iteration_size=self.getIterationSize(),
                              stride=self.getStride())
                self.setArray(array)

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)
