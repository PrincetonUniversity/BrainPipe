import numpy as np, tifffile
from soup.classic import *
from scipy.ndimage import median_filter, convolve
"""
http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html
http://vzaguskin.github.io/histmatching1/
https://en.wikipedia.org/wiki/Histogram_matching
"""

def im_cdf(im, maxx=2**16):
    """Equalizes image histogram, then returns cdf
    """
    bins = np.arange(0, maxx+1)
    hist,_ = np.histogram(im.flat, bins, density=True)
    cdf = hist.cumsum()
    cdf = maxx * cdf / cdf[-1]
    return cdf,bins

def hist_map(im, src, tgt):
    """Match histogram of src to that of tgt and use that mapping to transform im.

    for every value in target image, find the value in the source image whose CDF matches the CDF of that former value. this builds a mapping from values in the target to values in the source. then apply that mapping to your image
    """

    src_cdf,bins = im_cdf(src)
    tgt_cdf,bins = im_cdf(tgt)

    img_cdf = np.interp(im.flat, bins[:-1], src_cdf)
    im_transformed = np.interp(img_cdf, tgt_cdf, bins[:-1])

    return im_transformed

if __name__ == '__main__':
    # read in template
    target = tifffile.imread('target.tif')
    target = target[target>np.percentile(target, 5)]
    target = (2**16*(target-target.min())/(target.max()-target.min())).astype(np.uint16)

    # read in data
    im = tifffile.imread('brain.tif')

    # sqrt transformation
    im = np.sqrt(im)

    # match histogram to template
    mididx = len(im)//2
    for chan in range(im.shape[1]):
        mid = im[mididx, chan, :, :]
        chandata = im[:,chan,...]
        im[:,chan,...] = hist_map(chandata, mid, target).reshape(chandata.shape)

    # median filter, lowpass filter
    # lowpass params
    kernsize = 15
    kern = np.ones([kernsize, kernsize])/kernsize**2
    shape = im.shape
    im = im.reshape([im.shape[0]*im.shape[1],im.shape[2],im.shape[3]])
    for i in range(len(im)):
        im[i] = median_filter(im[i], 3)
        #im[i] = convolve(im[i], kern)
    im = im.reshape(shape)

    # sandbox
    af = im[:,1,...]
    sig = im[:,0,...]

