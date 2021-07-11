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

def to_eight_bit_range(img):
    imgcp = img.copy()
    imgcp[imgcp>np.percentile(img,99)] = np.percentile(img,99)
    if np.min(imgcp) < 0:
        imgcp=np.add(imgcp,abs(min(imgcp))
    if np.max(imgcp) > 255:
        imgcp = np.divide(imgcp,np.ceil(np.max(imgcp)/256))
    return imgcp

def resize_helper(img, dst, resizef):
    print(os.path.basename(img))
    im = sitk.GetArrayFromImage(sitk.ReadImage(img))
    y,x = im.shape
    yr = int(y/resizef); xr = int(x/resizef)
    im = cv2.resize(im, (xr, yr), interpolation=cv2.INTER_LINEAR)
    tif.imsave(os.path.join(dst, os.path.basename(img)), 
                    im.astype("uint16"), compress=1)


def get_folderstructure(dirname):
    folderstructure = []
    for i in os.walk(src):
        folderstructure.append(i)
    return folderstructure


def dwnsz(pth,save_str,src):
    print('starting dnsz fx')
    #now downsample to 140% of pra atlas
    imgs = [os.path.join(dst, xx) for xx in os.listdir(dst) if "tif" in xx]; imgs.sort()
    z = len(imgs)
    y,x = sitk.GetArrayFromImage(sitk.ReadImage(imgs[0])).shape
    arr = np.zeros((z,y,x))
    atlpth = "/jukebox/brody/ejdennis/lightsheet/mPRA.tif"
    atl = sitk.GetArrayFromImage(sitk.ReadImage(atlpth))
    atlz,atly,atlx = atl.shape #get shape, sagittal
    #read all the downsized images
    for i,img in enumerate(imgs):
        if i%5000==0: print(i)
        arr[i,:,:] = sitk.GetArrayFromImage(sitk.ReadImage(img)) #axial
    zz,yy,xx=arr.shape
    print("############### THE AXES ARE {},{},{}".format(zz,yy,xx))
    #switch to sagittal
    arrsag = np.swapaxes(arr,2,0)
    z,y,x = arrsag.shape
    print("############### THE NEW AXES ARE {},{},{}".format(z,y,x))
    print("\n**********downsizing....heavy!**********\n")
    arrsagd = zoom(arrsag, ((atlz*1.4/z),(atly*1.4/y),(atlx*1.4/x)), order=1)
    arrsagd = to_eight_bit_range(arrsagd)
    print('saving tiff at {}'.format(os.path.join(os.path.dirname(dst), "{}_downsized_for_atlas.tif".format(savestr))))
    tif.imsave(os.path.join(os.path.dirname(dst), "{}_downsized_for_atlas.tif".format(savestr)), arrsagd.astype("uint8"))


if __name__ == "__main__":
    
    #takes 3 command line args
    print(sys.argv)
    src=str(sys.argv[1]) #folder to main image folder
    rawdata=[]
    rawdata.append(os.path.join(src,str(sys.argv[2])))
    rawdata.append(os.path.join(src,str(sys.argv[3])))
    print(rawdata)
    for i in rawdata:
        if 'Ex_488' in i:
            reg_ch = i
            while len([file for file in os.listdir(reg_ch) if '.tif' in file]) < 10:
                reg_ch = os.path.join(reg_ch,[f.name for f in os.scandir(reg_ch) if f.is_dir()][0])
            print('reg ch is {}'.format(reg_ch))
            print("downsizing to {}".format(src))
            savestr="reg_"
            pth= reg_ch
        if 'Ex_64' in i:
            cell_ch=i
            while len([file for file in os.listdir(cell_ch) if '.tif' in file]) < 10:
       	       	cell_ch = os.path.join(cell_ch,[f.name for f in os.scandir(cell_ch) if f.is_dir()][0])
            print('cell ch is {}'.format(cell_ch))
            print("downsizing to {}".format(src))
            savestr = "cell_"
            pth=cell_ch

        print("path to atlas used for scaling {}".format(atlpth))
        print("\nPath to stitched images: %s\n" % pth)
        #path to store downsized images
        dst = os.path.join(os.path.dirname(src), "{}_downsized".format(savestr))
        print("\nPath to storage directory: %s\n\n" % dst)
        if not os.path.exists(dst): os.mkdir(dst)
        imgs = [os.path.join(pth, xx) for xx in os.listdir(pth) if "tif" in xx]
        z = len(imgs)
        imgtest = tif.imread(imgs[int(z/2)]))
        x,y=np.shape(imgtest)
        print("z is {}, y is {}, x is {}".format(z,y,x))
        np.save(os.path.join(src,"zyx.npy"),[z,y,x])
        resizef = 5 #factor to downsize imgs by
        print("resize factor is {}".format(resizef))
        iterlst = [(img, dst, resizef) for img in imgs]
        p = mp.Pool(11)
        p.starmap(resize_helper, iterlst)
        p.terminate()
        print("terminated p")
        dwnsz(pth,savestr,src)

    print('done')
