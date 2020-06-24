from collections import OrderedDict
import numpy as np


def to_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.squeeze(data, axis=0)
    else:
        raise RuntimeError("data must be a numpy 3D array")
    assert data.ndim==3
    return data


def to_tensor(data):
    """Ensure that data is a numpy 4D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError("data must be a numpy 4D array")
    assert data.ndim==4
    return data


def sort(sample):
    """Ensure that sample is sorted by key."""
    return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))
