# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:50:46 2016

@author: tpisano + bdeverett

Two approaches. Typically find_site is the more robust for general CTB injection detection. detect_inj_site has more parameters and might be better tunable for specific approaches
"""
from __future__ import division
import SimpleITK as sitk
import numpy as np
import cv2 #this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages
from matplotlib import pyplot as plt
from skimage.external import tifffile
from math import ceil
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer
from scipy.ndimage.filters import gaussian_filter as gfilt
from scipy.ndimage import label
#%%

#find_site
if __name__ == '__main__':
    im = tifffile.imread('/home/wanglab/wang/pisano/tracing_output/antero/20161201_db_bl6_lob6b_500r_53d5hr/20161201_db_bl6_lob6b_500r_53d5hr_488_555_647_z3um_250msec_1hfds_0005na_resized_ch01_resampledforelastix.tif')
    ctb = find_site(im)
    #n=np.copy(im)
    #n[ctb]=255
    #n[~ctb]=0
    #sitk.Show(sitk.GetImageFromArray(n))
    #plt.ion(); plt.figure; plt.imshow(np.max(n, axis=0))


#detect_inj_site
if __name__ == '__main__':
    image = '/home/wanglab/Desktop/b/l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_C01_Z1960.tif'
    disp, center, cnt = detect_inj_site(image, threshold = 150, kernelsize=20, minimumarea = 50000, xyz_scale=(1.63,1.63,3), testing=True)

#%%
def find_site(img, thresh=10, filter_kernel=(5,5,5)):
    """Find a connected area of high intensity, using a basic filter + threshold + connected components approach

    by: bdeverett Created on Fri May 19 14:46:55 2017

    Parameters
    ----------
    img : np.ndarray
        3D stack in which to find site (technically need not be 3D, so long as filter parameter is adjusted accordingly)
    thresh: float
        threshold for site-of-interest intensity, in number of standard deviations above the mean
    filter_kernel: tuple
        kernel for filtering of image before thresholding

    Returns
    --------
    bool array of volume where coordinates where detected
    """

    filtered = gfilt(im, filter_kernel)
    thresholded = filtered > filtered.mean() + thresh*filtered.std()
    labelled,nlab = label(thresholded)

    if nlab == 0:
        raise Exception('Site not detected, try a lower threshold?')
    elif nlab == 1:
        return labelled.astype(bool)
    else:
        sizes = [np.sum(labelled==i) for i in range(1,nlab+1)]
        return labelled == np.argmax(sizes)+1

#%%
def detect_inj_site(image, threshold = 150, kernelsize=20, minimumarea = 50000, xyz_scale=(1.63,1.63,3), testing=False):
    '''Function to combine binarize_erode_dilate and find_largest_contour

    Inputs:
    -----------
        image=path to file or numpy array
        threshold = pixel intensity theshold (of usually 16bit) to binarize (final output is 8bit)
        kernelsize = size of kernel for erosion and dilation, usually at least 5
        closing = image from binarize_erode_dilate function
        minimumarea = largest contour is below then function returns "None"
        xyz_scale: (x_microns_per_pixel, y..., z...); NOTE THIS IS USED TO CALCULATE KERNELSIZE & MINIMUM AREA THRESHOLDS, BUT OUTPUT IS IN PIXELS
        testing = using SITK+ImageJ function will display final binarized, eroded, and dilated image

    Returns
    -----------
        disp, center, cnt
    '''
    closing = binarize_erode_dilate(image, threshold = threshold, kernelsize = kernelsize, xyz_scale = xyz_scale, testing = testing)
    disp, center, cnt = find_largest_contour(closing, minimumarea = minimumarea, xyz_scale = xyz_scale, testing = testing)
    return disp, center, cnt #order is important

#%%
def binarize_erode_dilate(image, threshold = 150, kernelsize=12, xyz_scale=(1,1,1), testing = False):
    '''Function to load, binarize, erode and dilate image before passing to find_largest contour
    Inputs:
        image=path to file or numpy array
        threshold = pixel intensity theshold (of usually 16bit) to binarize (final output is 8bit)
        kernelsize = size of kernel for erosion and dilation, usually at least 5; this is calculated give xyz_scale
        xyz_scale: (x_microns_per_pixel, y..., z...); NOTE THIS IS USED TO CALCULATE KERNELSIZE, BUT OUTPUT IS IN PIXELS
        testing = using SITK+ImageJ function will display final binarized, eroded, and dilated image

    modified from http://creativemorphometrics.co.vu/blog/2014/08/05/automated-outlines-with-opencv-in-python/
    '''
    #load
    if type(image) == str:
        im = tifffile.imread(image)
    else:
        im = image.astype('uint8')
    #binarize
    thresh1=np.copy(im).astype('uint8')
    thresh1[im <= threshold] = 0; thresh1[im > threshold] = 255;
    #square image kernel used for erosion
    kernel = np.ones((int(ceil(kernelsize/xyz_scale[0])),int(ceil(kernelsize/xyz_scale[0]))), np.uint8)
    #refines all edges in the binary image
    erosion = cv2.erode(thresh1, kernel,iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image
    #optional output of image
    if testing == True:
        sitk.Show(sitk.GetImageFromArray(closing))
    return closing

def find_largest_contour(closing, minimumarea = None, xyz_scale=(1,1,1), testing = False):
    '''Function to find largest contour above some value.
    Inputs:
        closing = image from binarize_erode_dilate function
        minimumarea = largest contour is below then function returns "None"
        xyz_scale: (x_microns_per_pixel, y..., z...); NOTE THIS IS USED TO CALCULATE MINIMUM AREA THRESHOLDS, BUT OUTPUT IS IN PIXELS
        testing = using SITK+ImageJ function will display 1) all contours, 2) 'max' contour


    modified from http://creativemorphometrics.co.vu/blog/2014/08/05/automated-outlines-with-opencv-in-python/
    '''
    #find contours with simple approximation
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(closing, contours, -1, (255, 255, 255), 4)
    areas = [] #list to hold all areas
    #calculate areas
    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)

    try:
        ##find max
        max_area = max(areas)
        max_area_index = areas.index(max_area) #index of the list element with largest area
        #set cnt for largest cnt
        cnt = np.squeeze(contours[max_area_index]) #largest area contour

        cv2.drawContours(closing, [cnt], 0, (255, 255, 255), 3, maxLevel = 0)
    except ValueError:
        max_area=0
        cnt=np.asarray(np.zeros((0,0)))

    #generate display image of max contour
    disp=np.zeros(closing.shape); cv2.drawContours(disp, [cnt], 0, (255, 255, 255), 3, maxLevel = 0)

    #if max_area not greater than minimum area return nothing
    if minimumarea != None:
        if max_area < (minimumarea / (xyz_scale[0] * xyz_scale[1])):
            cnt = None ###set cnt to None
            center = None
            return disp, center, cnt
    #find center of cnt using mean
    center = np.asarray(np.mean(cnt, axis=0))

    ###output control
    if testing != False:
        sitk.Show(sitk.GetImageFromArray(closing))  ##return all contours
        sitk.Show(sitk.GetImageFromArray(disp)) ##return max contour

    return disp, center, cnt
