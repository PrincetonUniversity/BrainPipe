#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:11:57 2020

@author: wanglab
"""

import os, numpy as np, tifffile as tif, SimpleITK as sitk, cv2, multiprocessing as mp, shutil, sys
import pandas as pd
from scipy.ndimage import zoom
from subprocess import check_output

def csv_to_dict(csv_pth):
    """ 
    reads csv and converts to dictionary
    1st column = keys
    2nd column = values
    """
    csv_dict = {}
    
    params = pd.read_csv(csv_pth, header = None)
    keys = list(params[0])
    for i,val in enumerate(params[1].values):
        try:
            csv_dict[keys[i]] = ast.literal_eval(val)
        except:
            csv_dict[keys[i]] = val
            
    return csv_dict

def sp_call(call):
    print(check_output(call, shell=True))
    return 

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
    
def stitch(volin,xy,z,dest):
    """ call function for terastitcher """
    sp_cell("terastitcher -1 --volin=%s --ref1=x --ref2=y --ref3=z --vxl1=%d --vxl2=%d --vxl3=%d --projout=xml_import" % (volin,xy,xy,z))
    sp_call("terastitcher --displcompute --projin=%s --subvoldim=100 --projout=xml_displcomp" % os.path.join(volin, "xml_import.xml"))
    sp_call("terastitcher --displproj --projin=%s" % (os.path.join(volin, "xml_displcomp.xml")))
    sp_call("terastitcher --displthres --projin=%s --projout=%s --threshold=0.7" % (os.path.join(volin, "xml_displproj.xml"),os.path.join(volin, "xml_displthres.xml")))
    sp_call("terastitcher --placetiles --projin=%s --projout=%s --algorithm=MST" % (os.path.join(volin, "xml_displthres.xml"),os.path.join(volin, "xml_placetiles.xml")))
    sp_call("terastitcher --merge --projin=%s --volout=%s --imout_depth=16 --resolutions=0" % (os.path.join(volin, "xml_placetiles.xml"), dest))
    
    return dest

def downsize(volin,resizefactor,cores,atl):
    """ downsize full resolution data """
    #path to store downsized images
    dst = os.path.join(volin, "downsized")
    print("\npath to storage directory: %s\n\n" % dst)
    if not os.path.exists(dst): os.mkdir(dst)
    imgs = [os.path.join(volin, xx) for xx in os.listdir(volin) if "tif" in xx]
    z = len(imgs)
    iterlst = [(img, dst, resizefactor) for img in imgs]
    p = mp.Pool(cores)
    p.starmap(resize_helper, iterlst)
    p.terminate()
    
    #now downsample to 140% of pma atlas
    imgs = [os.path.join(dst, xx) for xx in os.listdir(dst) if "tif" in xx]; imgs.sort()
    z = len(imgs)
    y,x = sitk.GetArrayFromImage(sitk.ReadImage(imgs[0])).shape
    arr = np.zeros((z,y,x))
    atl = sitk.GetArrayFromImage(sitk.ReadImage(atl))
    atlz,atly,atlx = atl.shape #get shape, sagittal
    #read all the downsized images
    for i,img in enumerate(imgs):
        if i%5000==0: print(i)
        arr[i,:,:] = sitk.GetArrayFromImage(sitk.ReadImage(img)) #horizontal
    #switch to specified orientation
    if orientation == "sagittal":
        arrsag = np.swapaxes(arr,2,0)
    z,y,x = arrsag.shape
    print((z,y,x))
    print("\n**********downsizing....heavy!**********\n")
    
    arrsagd = zoom(arrsag, ((atlz*1.4/z),(atly*1.4/y),(atlx*1.4/x)), order=1)
    tif.imsave(os.path.join(os.path.dirname(dst), "downsized_for_atlas.tif"), arrsagd.astype("uint16"))
    print("\ndeleting storage directory after making volume...\n %s" % dst)
    shutil.rmtree(dst)
    return os.path.join(os.path.dirname(dst), "downsized_for_atlas.tif")

def register(channel1_downsized,atl,parameterfld,channel2_downsized=None,channel3_downsized=None,
             channel4_downsized=None):
    """ perform registration and inverse registration """
    
    mv = channel1_downsized
    print("\npath to downsized vol for registration to atlas: %s" % mv)
    fx = atl
    print("\npath to atlas: %s" % fx)
    out = os.path.join(os.path.dirname(src), "elastix")
    if not os.path.exists(out): os.mkdir(out)
    
    params = [os.path.join(parameterfld, xx) for xx in os.listdir(parameterfld)]
    #run
    e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)

    if channel2_downsized:
        #cell vol to registration vol
        print("\nCell channel specified: %s" % cell)
        mv = os.path.join(src, cell+"/downsized_for_atlas.tif")
        fx = os.path.join(src, reg+"/downsized_for_atlas.tif")
        
        out = os.path.join(src, "elastix/%s_to_%s" % (cell, reg))
        if not os.path.exists(out): os.mkdir(out)
        
        params = [os.path.join(param_fld, xx) for xx in os.listdir(param_fld)]
        #run
        e_out, transformfiles = elastix_command_line_call(fx, mv, out, params)