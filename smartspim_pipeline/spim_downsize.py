#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:04:02 2020

@author: wanglab
"""

import os, numpy as np, tifffile as tif, SimpleITK as sitk, cv2, multiprocessing as mp, shutil, sys
from scipy.ndimage import zoom

def fast_scandir(dirname):
    """ gets all folders recursively """
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def resize_helper(img, dst, resizef):
    print(os.path.basename(img))
    im = sitk.GetArrayFromImage(sitk.ReadImage(img))
    y,x = im.shape
    yr = int(y/resizef); xr = int(x/resizef)
    im = cv2.resize(im, (xr, yr), interpolation=cv2.INTER_LINEAR)
    tif.imsave(os.path.join(dst, os.path.basename(img)), 
                    im.astype("uint16"), compress=1)

if __name__ == "__main__":
    
    #takes 1 command line args
    print(sys.argv)
    src=str(sys.argv[1]) #folder to main image folder
    pth = fast_scandir(src)[-1] #gets to the end of directory tree
    print("\nPath to stitched images: %s\n" % pth)
    #path to store downsized images
    dst = os.path.join(os.path.dirname(src), "downsized")
    print("\nPath to storage directory: %s\n\n" % dst)
    if not os.path.exists(dst): os.mkdir(dst)
    imgs = [os.path.join(pth, xx) for xx in os.listdir(pth) if "tif" in xx]
    z = len(imgs)
    resizef = 5 #factor to downsize imgs by
    iterlst = [(img, dst, resizef) for img in imgs]
    p = mp.Pool(12)
    p.starmap(resize_helper, iterlst)
    p.terminate()
    
    #now downsample to 140% of pma atlas
    imgs = [os.path.join(dst, xx) for xx in os.listdir(dst) if "tif" in xx]; imgs.sort()
    z = len(imgs)
    y,x = sitk.GetArrayFromImage(sitk.ReadImage(imgs[0])).shape
    arr = np.zeros((z,y,x))
    atlpth = "/jukebox/LightSheetTransfer/atlas/allen_atlas/average_template_25_sagittal_forDVscans.tif"
    atl = sitk.GetArrayFromImage(sitk.ReadImage(atlpth))
    atlz,atly,atlx = atl.shape #get shape, sagittal
    #read all the downsized images
    for i,img in enumerate(imgs):
        if i%5000==0: print(i)
        arr[i,:,:] = sitk.GetArrayFromImage(sitk.ReadImage(img)) #horizontal
    #switch to sagittal
    arrsag = np.swapaxes(arr,2,0)
    z,y,x = arrsag.shape
    print((z,y,x))
    print("\n**********downsizing....heavy!**********\n")
    
    arrsagd = zoom(arrsag, ((atlz*1.4/z),(atly*1.4/y),(atlx*1.4/x)), order=1)
    tif.imsave(os.path.join(os.path.dirname(dst), "downsized_for_atlas.tif"), arrsagd.astype("uint16"))
    print("\ndeleting storage directory after making volume...\n %s" % dst)
    shutil.rmtree(dst)
