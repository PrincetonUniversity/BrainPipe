from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Perturb(object):
    """
    Callable class for in-place image perturbation.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Grayscale(Perturb):
    """Grayscale intensity perturbation."""
    def __init__(self, contrast_factor=0.3, brightness_factor=0.3):
        contrast_factor = np.clip(contrast_factor, 0, 2)
        brightness_factor = np.clip(brightness_factor, 0, 2)
        params = dict()
        params['contrast'] = 1 + (np.random.rand() - 0.5) * contrast_factor
        params['brightness'] = (np.random.rand() - 0.5) * brightness_factor
        params['gamma'] = (np.random.rand()*2 - 1)
        self.params = params

    def __call__(self, img):
        img *= self.params['contrast']
        img += self.params['brightness']
        np.clip(img, 0, 1, out=img)
        img **= 2.0**self.params['gamma']

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'contrast={:.2f}, '.format(self.params['contrast'])
        format_string += 'brightness={:.2f}, '.format(self.params['brightness'])
        format_string += 'gamma={:.2f}'.format(self.params['gamma'])
        format_string += ')'
        return format_string


class Fill(Perturb):
    """Fill with a scalar."""
    def __init__(self, value=0, random=False):
        value = np.clip(value, 0, 1)
        self.value = np.random.rand() if random else value

    def __call__(self, img):
        img[...] = self.value

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'value={:.3f}'.format(self.value)
        format_string += ')'
        return format_string


class Blur(Perturb):
    """Gaussian blurring."""
    def __init__(self, sigma=5.0, random=False):
        sigma = max(sigma, 0)
        self.sigma = np.random.rand()*sigma if random else sigma

    def __call__(self, img):
        gaussian_filter(img, sigma=self.sigma, output=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma={:.2f}'.format(self.sigma)
        format_string += ')'
        return format_string


class Blur3D(Perturb):
    """Gaussian blurring."""
    def __init__(self, sigma=(5.0,5.0,5.0), random=False):
        self.sigma = [np.random.rand()*s for s in sigma] if random else sigma

    def __call__(self, img):
        gaussian_filter(img, sigma=self.sigma, output=img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma=({:.2f},{:.2f},{:.2f})'.format(*self.sigma)
        format_string += ')'
        return format_string


class Noise(Perturb):
    """Uniform noise + Gaussian blurring."""
    def __init__(self, sigma=(2,5)):
        assert len(sigma)==2
        self.sigma = tuple(max(s, 0) for s in sigma)

    def __call__(self, img):
        patch = (np.random.rand(*img.shape[-3:])).astype(img.dtype)
        s1 = self.sigma[0]
        gaussian_filter(patch, sigma=(0,s1,s1), output=patch)
        patch = (patch > 0.5).astype(img.dtype)
        s2 = self.sigma[1]
        gaussian_filter(patch, sigma=(0,s2,s2), output=patch)
        img[...,:,:,:] = patch

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma={:}'.format(self.sigma)
        format_string += ')'
        return format_string
