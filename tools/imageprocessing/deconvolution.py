#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:59:42 2017

@author: tpisano
"""

#RL deconvolution
#http://scikit-image.org/docs/dev/auto_examples/filters/plot_deconvolution.html
pth = '/home/wanglab/wang/pisano/Python/psf_for_deconvolution/psf_001na_650_665_5umx3um_lswidth15um.h5'
import h5py as hp, numpy as np
fl = hp.File(pth, 'r')
psf = np.max(fl['psf']['ImageData']['Image'][0,0], axis=0)#[28:37, 28:37]

from skimage.external import tifffile
im = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero/20161214_db_bl6_lob7_left_53hrs/full_sizedatafld/20161214_db_bl6_lob7_left_53hrs_488_555_647_0005na_1hfsds_z3um_250msec_ch02/20161214_db_bl6_lob7_left_53hrs_488_555_647_0005na_1hfsds_z3um_250msec_C02_Z1734.tif')

from skimage import color, data, restoration
deconvolved_RL = restoration.richardson_lucy(im, psf, iterations=3)
weiner = restoration.wiener(im, psf, 1100)

import matplotlib.pyplot as plt, numpy as np

plt.ion(); plt.figure(); plt.imshow(im)
plt.figure(); plt.imshow(deconvolved_RL)
plt.figure(); plt.imshow(weiner)




#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

astro = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
       a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()
