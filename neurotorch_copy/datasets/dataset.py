from torch.utils.data import Dataset as _Dataset
import numpy as np
from abc import abstractmethod
from neurotorch.datasets.datatypes import BoundingBox, Vector
from numbers import Number
from numpy import ndarray
from scipy.spatial import KDTree
from functools import reduce


class Data:
    """
    An encapsulating object for communicating volumetric data
    """
    def __init__(self, array: ndarray, bounding_box: BoundingBox):
        """
        Initializes a data packet from an Numpy array and its bounding box

        :param array: A Numpy array containing the data packet's contents
        :param bounding_box: A bounding box specifying the data packet's
location in 3D-space
        """
        self._setBoundingBox(bounding_box)
        self._setArray(array)

    def getBoundingBox(self) -> BoundingBox:
        """
        Returns the bounding box specifying the data packet's location in
3D-space

        :return: The bounding box of the data packet's location
        """
        return self.bounding_box

    def _setBoundingBox(self, bounding_box: BoundingBox):
        """
        Set the bounding box specifying the data packet's location in 3D-space

        :param bounding_box: The bounding box of the data packet's location
        """
        if not isinstance(bounding_box, BoundingBox):
            raise ValueError("bounding_box must have type BoundingBox")

        self.bounding_box = bounding_box

    def getArray(self) -> ndarray:
        """
        Retrieves the data packet's contents

        :return: An Numpy ndarray in row-major order (Z, Y, X)
        """
        return self.array

    def _setArray(self, array):
        """
        Sets the data packet's contents

        :param array: An Numpy ndarray in row-major order (Z, Y, X)
        """
        self.array = array

    def getSize(self) -> Vector:
        """
        Retrieves the size of the data packet

        :return: A vector containing the data packet's size
        """
        return self.getBoundingBox().getSize()

    def __add__(self, other):
        if not isinstance(other, Data):
            raise ValueError("other must have type Data")
        if self.getBoundingBox() != other.getBoundingBox():
            raise ValueError("other must have the same bounding box")

        return Data(self.getArray() + other.getArray(),
                    self.getBoundingBox())

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return (self * -1)

    def __mul__(self, other):
        if not isinstance(other, Number):
            error_string = "other must be a number instead it is a {}"
            error_string = error_string.format(type(other))
            raise ValueError(error_string)

        return Data(self.getArray() * other,
                    self.getBoundingBox())

    def __div__(self, other):
        if not isinstance(other, Number):
            error_string = "other must be a number instead it is a {}"
            error_string = error_string.format(type(other))
            raise ValueError(error_string)

        return (self * (1/other))


class Array:
    """
    A dataset containing a 3D volumetric array
    """
    def __init__(self, array: np.ndarray, bounding_box: BoundingBox=None,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Initializes a volume with a bounding box and iteration parameters

        :param array: A 3D Numpy array
        :param bounding_box: The bounding box encompassing the volume
        :param iteration_size: The bounding box of each data sample in the
dataset iterable
        :param stride: The stride displacement of each data sample in the
dataset iterable. The displacement proceeds first from X then to Y then to Z.
        """
        if isinstance(array, np.ndarray):
            self._setArray(array)
        elif isinstance(array, BoundingBox):
            self.createArray(array)
        else:
            raise ValueError("array must be an ndarray or a BoundingBox")

        self.setBoundingBox(bounding_box)
        self.setIteration(iteration_size=iteration_size,
                          stride=stride)
        super().__init__()

    def get(self, bounding_box: BoundingBox) -> Data:
        """
        Requests a data sample from the volume. If the bounding box does
not exist, then the method raises a ValueError.

        :param bounding_box: The bounding box of the request data sample
        :return: The data sample requested
        """
        if bounding_box.isDisjoint(self.getBoundingBox()):
            error_string = ("Bounding box must be inside dataset " +
                            "dimensions instead bounding box is {} while " +
                            "the dataset dimensions are {}")
            error_string = error_string.format(bounding_box,
                                               self.getBoundingBox())
            raise ValueError(error_string)

        sub_bounding_box = bounding_box.intersect(self.getBoundingBox())
        array = self.getArray(sub_bounding_box)

        before_pad = bounding_box.getEdges()[0] - sub_bounding_box.getEdges()[0]
        after_pad = sub_bounding_box.getEdges()[1] - bounding_box.getEdges()[1]

        if before_pad != Vector(0, 0, 0) and after_pad != Vector(0, 0, 0):
            pad_size = (before_pad.getComponents(),
                        after_pad.getComponents())
            array = np.pad(array, pad_size=pad_size, mode="constant")

        return Data(array, bounding_box)

    def set(self, data: Data):
        """
        Sets a section of the volume within the provided bounding box with the
given data.

        :param data: The data packet to set the volume
        """
        data_bounding_box = data.getBoundingBox()
        data_array = data.getArray()

        if not data_bounding_box.isSubset(self.getBoundingBox()):
            raise ValueError("The bounding box must be a subset of the "
                             " volume")

        data_edge1, data_edge2 = data_bounding_box.getEdges()
        array_edge1, array_edge2 = self.getBoundingBox().getEdges()

        edge1 = data_edge1 - array_edge1
        edge2 = data_edge2 - array_edge1

        x1, y1, z1 = edge1.getComponents()
        x2, y2, z2 = edge2.getComponents()

        self.array[z1:z2, y1:y2, x1:x2] = data_array

    def blend(self, data: Data):
        """
        Blends a section of the volume within the provided bounding box with
the given data by taking the elementwise maximum value.

        :param data: The data packet to blend into the volume
        """
        array = self.get(data.getBoundingBox()).getArray()
        array = np.maximum(array, data.getArray())

        result = Data(array, data.getBoundingBox())

        self.set(result)

    def getArray(self, bounding_box: BoundingBox=None) -> np.ndarray:
        """
        Retrieves the array contents of the volume. If a bounding box is
provided, the subsection is returned.

        :param bounding_box: The bounding box of a subsection of the volume.
If the bounding box is outside of the volume, a ValueError is raised.
        """
        if bounding_box is None:
            return self.array

        else:
            if not bounding_box.isSubset(self.getBoundingBox()):
                raise ValueError("The bounding box must be a subset" +
                                 " of the volume")

            centered_bounding_box = bounding_box - self.getBoundingBox().getEdges()[0]
            edge1, edge2 = centered_bounding_box.getEdges()
            x1, y1, z1 = edge1.getComponents()
            x2, y2, z2 = edge2.getComponents()

            return self.array[z1:z2, y1:y2, x1:x2]

    def _setArray(self, array):
        self.array = array

    def getBoundingBox(self) -> BoundingBox:
        """
        Retrieves the bounding box of the volume

        :return: The bounding box of the volume
        """
        return self.bounding_box

    def setBoundingBox(self, bounding_box: BoundingBox=None,
                       displacement: Vector=None):
        """
        Sets the bounding box of the volume. By default, it sets the bounding
box to the volume size

        :param bounding_box: The bounding box of the volume
        :param displacement: The displacement of the bounding box from the
origin
        """
        if bounding_box is None:
            self.bounding_box = BoundingBox(Vector(0, 0, 0),
                                            Vector(*self.getArray().shape[::-1]))
        else:
            self.bounding_box = bounding_box

        if displacement is not None:
            self.bounding_box = self.bounding_box + displacement

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        """
        Sets the parameters for iterating through the dataset

        :param iteration_size: The size of each data sample in the volume
        :param stride: The displacement of each iteration
        """
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

    def __enter__(self):
        pass

    def __exit__(self):
        pass


class TorchVolume(_Dataset):
    def __init__(self, volume):
        self.setVolume(volume)
        super().__init__()

    def __len__(self):
        return len(self.getVolume())

    def __getitem__(self, idx):
        if isinstance(self.getVolume(), AlignedVolume):
            data_list = [self.toTorch(data) for data in self.getVolume()[idx]]
            return data_list
        else:
            return self.getVolume()[idx].getArray()

    def toTorch(self, data):
        torch_data = data.getArray().astype(np.float)
        torch_data = torch_data.reshape(1, *torch_data.shape)
        return torch_data

    def setVolume(self, volume):
        self.volume = volume

    def getVolume(self):
        return self.volume


class Volume:
    """
    An interface for creating volumes
    """
    def __init__(self, bounding_box: BoundingBox=None,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        self.setBoundingBox(bounding_box)
        self.setIterationSize(iteration_size)
        self.setStride(stride)

    def setArray(self, array: Array):
        self.array = array

    def getArray(self) -> Array:
        return self.array

    def request(self, bounding_box):
        return self.getArray().get(bounding_box)

    def set(self, data: Data):
        self.getArray().set(data)

    def blend(self, data: Data):
        self.getArray().blend(data)

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        self.getArray().setIteration(iteration_size, stride)
        self.setIterationSize(iteration_size)
        self.setStride(stride)

    def setBoundingBox(self, bounding_box):
        if not isinstance(bounding_box, BoundingBox):
            raise ValueError("bounding_box must have type BoundingBox " +
                             "instead it has type {}".format(type(bounding_box)))
        self.bounding_box = bounding_box

    def getBoundingBox(self):
        return self.bounding_box

    def setIterationSize(self, iteration_size):
        if not isinstance(iteration_size, BoundingBox):
            raise ValueError("iteration_size must have type BoundingBox")
        self.iteration_size = iteration_size

    def getIterationSize(self):
        return self.iteration_size

    def setStride(self, stride):
        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")
        self.stride = stride

    def getStride(self):
        return self.stride

    @abstractmethod
    def loadArray(self):
        pass

    @abstractmethod
    def unloadArray(self):
        pass

    @abstractmethod
    def get(self, bounding_box: BoundingBox) -> Data:
        """
        Requests a data sample from the dataset. If the bounding box does
not exist, then the method raises a ValueError.

        :param bounding_box: The bounding box of the request data sample
        :return: The data sample requested
        """
        pass

    @abstractmethod
    def set(self, data: Data):
        """
        Sets a section of the dataset within the provided bounding box with the
given data.

        :param data: The data packet to set the volume
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Loads the dataset into memory
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Unloads the dataset from memory
        """
        pass

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: The dataset length
        """
        return len(self.getArray())

    def __getitem__(self, idx: int):
        """
        Returns the data sample at index idx from the dataset

        :param idx: The index of the data sample
        """
        return self.getArray()[idx]

    def __iter__(self):
        """
        Returns an iterable of the dataset

        :return: The iterable of the dataset
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Retrieves the next data sample from the dataset
        :return: The next data sample
        """
        if self.index < len(self):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration


class AlignedVolume(Array):
    def __init__(self, volumes, iteration_size=None, stride=None):
        if iteration_size is None:
            iteration_size = volumes[0].getIterationSize()
        if stride is None:
            stride = volumes[0].getStride()
        self.setVolumes(volumes)
        self.setIteration(iteration_size, stride)

    def getBoundingBox(self):
        return self.getVolumes()[0].getBoundingBox()

    def setVolumes(self, volumes):
        self.volumes = volumes

    def addVolume(self, volume):
        self.volumes.append(volume)

    def getVolumes(self):
        return self.volumes

    def setIteration(self, iteration_size, stride):
        for volume in self.getVolumes():
            volume.setIteration(iteration_size, stride)

    def get(self, bounding_box):
        result = [volume.get(bounding_box)
                  for volume in self.getVolumes()]
        return result

    def set(self, array, bounding_box):
        pass

    def __len__(self):
        return len(self.getVolumes()[0])

    def __getitem__(self, idx):
        result = [volume[idx] for volume in self.getVolumes()]
        return result


class PooledVolume(Volume):
    def __init__(self, volumes=None, stack_size: int=5):
        if volumes is not None:
            self.volumes = volumes
            self.volumes_changed = True
        else:
            self.volumes = []
            self.volumes_changed = False

        self.volume_list = []
        self.setStack(stack_size)

    def setStack(self, stack_size: int=5):
        self.stack = []
        self.stack_size = 5

    def _pushStack(self, index, volume):
        if len(self.stack) >= self.stack_size:
            self.stack[0].__exit__(None, None, None)
            self.stack.pop(0)

        pos = len(self.stack)
        self.stack.insert(pos, (index, volume.__enter__()))

        return pos

    def _rebuildIndexes(self):
        edge1_list = [volume.getBoundingBox().getEdges()[0].getComponents()
                      for volume in self.volumes]

        self.edge1_list = KDTree(edge1_list)

        self.volumes_changed = False

    def _queryBoundingBox(self, bounding_box: BoundingBox) -> Volume:
        if self.volumes_changed:
            self._rebuildIndexes()

        edge1 = [bounding_box.getEdges()[0].getComponents()]
        distances, indexes  = self.edge1_list.query(edge1, k=8)
        indexes = [index for index, dist in zip(indexes[0], distances[0])
                   if dist < float('Inf')]
        indexes = filter(lambda index: not bounding_box.isDisjoint(self.volumes[index].getBoundingBox()),
                         indexes)
        if not indexes:
            raise IndexError("bounding_box is not present in any indexes")

        return list(indexes)

    def add(self, volume: Volume):
        self.volumes_changed = True
        self.volumes.append(volume)

    def get(self, bounding_box: BoundingBox) -> Data:
        indexes = self._queryBoundingBox(bounding_box)

        data = []

        stack_volumes = [volume for i, volume in self.stack if i in indexes]
        stack_disjoint = list(set(indexes) - set([i for i, v in self.stack]))

        for volume in stack_volumes:
            sub_bbox = bounding_box.intersect(volume.getBoundingBox())
            data.append(volume.request(sub_bbox))

        for index in stack_disjoint:
            volume = self.volumes[index]
            i = self._pushStack(index, volume)

            sub_bbox = bounding_box.intersect(volume.getBoundingBox())
            data.append(volume.request(sub_bbox))

        if len(data) > 1:
            shape = bounding_box.getNumpyDim()
            array = Array(np.zeros(shape).astype(np.uint16), bounding_box=bounding_box)
            [array.set(item) for item in data]
            return Data(array.getArray(), bounding_box)
        else:
            return data[0]

    def set(self, data: Data):
        indexes = self._queryBoundingBox(data.getBoundingBox())

        data = []
        for index in indexes:
            for stack_index, stack_volume in self.stack:
                if stack_index == index:
                    stack_volume.set(data)
                else:
                    volume = self.volume_list[index].__enter__()
                    self._pushStack(index, volume)

                    volume.set(data)

    def __exit__(self, exc_type, exc_value, traceback):
        for index, volume in self.stack:
            volume.__exit__()

    def __len__(self) -> int:
        if self.volumes_changed:
            self.volume_index = list(map(lambda volume: len(volume), self.volumes))
            for index, length in enumerate(self.volume_index):
                if index > 0:
                    self.volume_index[index] += self.volume_index[index-1]
            length = reduce(lambda x, y: x + y, self.volume_index)

        return length

#    def __getitem__(self, idx: int) -> Data:
#        index = 0
#        while self.volume_index[index] < idx:
#            index += 1
#        index += -1
#        _idx = idx-self.volume_index[index]
#        return self.volumes[index][_idx]
    
    def __getitem__(self, idx: int) -> Data:
        index = 0
        while self.volume_index[index]-1 < idx:
            index += 1
            _idx = idx - self.volume_index[index-1]
        
        _idx = idx
#        index += -1

        return self.volumes[index][_idx]
