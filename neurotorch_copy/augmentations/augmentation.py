from neurotorch.datasets.dataset import AlignedVolume
from neurotorch.datasets.datatypes import Vector
from abc import abstractmethod
import numpy as np


class Augmentation(AlignedVolume):
    def __init__(self, aligned_volume, iteration_size=None, stride=None):
        self.setVolume(aligned_volume)
        if iteration_size is None:
            iteration_size = self.getInputVolume().getIterationSize()
        if stride is None:
            stride = self.getInputVolume().getStride()
        self.setIteration(iteration_size, stride)

    def get(self, bounding_box):
        augmented_data = self.augment(bounding_box)
        return augmented_data

    def getBoundingBox(self):
        return self.getVolume().getBoundingBox()

    def setIteration(self, iteration_size, stride):
        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return int(round(x))

        self.element_vec = Vector(*map(lambda L, l, s: ceil((L-l)/s+1),
                                       self.getBoundingBox().getEdges()[1].getComponents(),
                                       self.iteration_size.getEdges()[1].getComponents(),
                                       self.stride.getComponents()))

        self.index = 0

    def getInputVolume(self):
        return self.getVolume().getVolumes()[0]

    def getLabelVolume(self):
        return self.getVolume().getVolumes()[1]

    def getInput(self, bounding_box):
        return self.getInputVolume().request(bounding_box)

    def getLabel(self, bounding_box):
        return self.getLabelVolume().request(bounding_box)

    def setIterationSize(self, iteration_size):
        self.iteration_size = iteration_size

    def setStride(self, stride):
        self.stride = stride

    def setVolume(self, aligned_volume):
        self.aligned_volume = aligned_volume

    def getVolume(self):
        return self.aligned_volume

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

    @abstractmethod
    def augment(self, bounding_box):
        pass
