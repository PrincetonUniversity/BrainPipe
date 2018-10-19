from neurotorch.augmentations.augmentation import Augmentation
from neurotorch.datasets.dataset import Data
import random
import numpy as np
from scipy.sparse import dok_matrix
from scipy.ndimage.filters import gaussian_filter


class Occlusion(Augmentation):
    def __init__(self, volume, frequency=0.1):
        self.setFrequency(frequency)
        super().__init__(volume)

    def augment(self, bounding_box):
        raw = self.getInput(bounding_box)
        label = self.getLabel(bounding_box)
        augmented_raw, augmented_label = self.occlude(raw, label)

        return (augmented_raw, augmented_label)

    def setFrequency(self, frequency):
        self.frequency = frequency

    def occlude(self, raw_data, label_data, size=(4, 20, 20)):
        # Get array representations from data
        raw = raw_data.getArray()
        label = label_data.getArray()

        # Find a random neuron voxel
        neuron = self.dok_volume(label)
        neuron_voxel = random.choice(list(neuron))

        # Get occluding region
        min_x = max(0, neuron_voxel[2]-50)
        min_y = max(0, neuron_voxel[1]-50)
        min_z = max(0, neuron_voxel[0]-8)

        max_x = min(raw.shape[2], neuron_voxel[2]+50)
        max_y = min(raw.shape[1], neuron_voxel[1]+50)
        max_z = min(raw.shape[0], neuron_voxel[0]+8)

        x_len = max_x-min_x
        y_len = max_y-min_y
        z_len = max_z-min_z

        psf = np.zeros((z_len, y_len, x_len))
        psf[z_len//2, y_len//2, x_len//2] = 1
        psf = gaussian_filter(psf, size)
        psf = psf/psf[z_len//2, y_len//2, x_len//2]

        # Get background statistics
        sub_raw = raw[min_z:max_z, min_y:max_y, min_x:max_x]
        sub_label = label[min_z:max_z, min_y:max_y, min_x:max_x].astype(np.bool)
        average = np.mean(sub_raw[~sub_label])
        stdev = np.std(sub_raw[~sub_label])

        # Occlude region
        filtered_raw = raw.copy()
        filtered_raw[min_z:max_z, min_y:max_y, min_x:max_x] = raw[min_z:max_z, min_y:max_y, min_x:max_x]*(1-psf) + \
                                                psf*np.random.normal(loc=average,
                                                                    scale=stdev,
                                                                    size=(z_len,
                                                                          y_len,
                                                                          x_len)).astype(np.uint16)

        augmented_raw_data = Data(filtered_raw, raw_data.getBoundingBox())

        return augmented_raw_data, label_data

    def dok_volume(self, volume):
        dok = []
        for index in range(volume.shape[0]):
            dok += [(index, point[0], point[1]) for point in list(dok_matrix(volume[index, :, :]).keys())]
        return dok
