#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:39:16 2018

@author: wanglab
"""

import os, numpy as np, sys, multiprocessing as mp, zipfile, sys
from skimage.external import tifffile
from skimage import filters
import matplotlib.pyplot as plt
import SimpleITK as sitk
#add your clone of the brainpipe repo to path to import the relevant modules
sys.path.append("/jukebox/wang/zahra/python/BrainPipe") 
from tools.conv_net.utils.io import listdirfull, load_np, makedir, save_dictionary

def otsu_par(saveLocation, otsufld, size, otsu_factor):
   
    #otsu
    p = mp.Pool(12)
    iterlst = [(otsufld, inn, size, otsu_factor) for inn in listdirfull(saveLocation, "npy")]
    p.starmap(otsu_helper, iterlst)
    p.terminate()
    return
            
def otsu_helper(otsufld, inn, size, otsu_factor):
    
    #load
    arr = load_np(inn)
    raw = np.copy(arr[0])
    lbl = np.copy(arr[1])
    
    #save input
    tifffile.imsave(os.path.join(otsufld, "{}_img.tif".format(os.path.basename(inn)[:-4])), raw.astype("float32"))
    
    #save input
    otsu = otsu_dilate(raw, lbl, size=size, otsu_factor=otsu_factor)
    tifffile.imsave(os.path.join(otsufld, "{}_lbl.tif".format(os.path.basename(inn)[:-4])), otsu.astype("float32"))

    print(inn)
    
    return

def otsu_dilate(arr0, arr1, size=(8,60,60), otsu_factor=0.8):
    """4d arr
    arr0=raw data
    arr1=points
    size=(z,y,x)
    otsu_factor - scaling of the otsu value, >1 is less stringent, <1 remove more pixels
    """
    #get points
    pnts = np.asarray(np.nonzero(arr1)).T.astype("int64")
    outarr = np.zeros_like(arr1)
    
    for pnt in pnts:
        #print pnt
        vol = np.copy(arr0[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]])*1.0
        #vol = filters.gaussian(vol, 1)
        v=filters.threshold_otsu(vol)/float(otsu_factor)
        vol[vol<v]=0
        vol[vol>=v]=1
        nvol = np.maximum(outarr[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]], vol)
        outarr[np.max((pnt[0]-size[0],0)):pnt[0]+size[0], np.max((pnt[1]-size[1],0)):pnt[1]+size[1], np.max((pnt[2]-size[2],0)):pnt[2]+size[2]]=nvol
    
    return outarr

def convert_input(inputFolder, saveLocation, remove_bad=True):
    """Function for converting data from imageJ ROIs + data to mem_mapped arrays for preprocessing + batch generation to pass to cnn
    """
    #get pairs
    tfs = listdirfull(inputFolder,keyword=".tif")
    zps = [xx for xx in listdirfull(inputFolder) if ".tif" not in xx]
    
    #make empty zip files if no labels (useful to train on negative data?)
    for tf in tfs:
        if tf[:-4]+"RoiSet.zip" not in zps:
            print(tf)
            nm = tf[:-4]+"RoiSet.zip"    
            with zipfile.ZipFile(os.path.join(inputFolder, nm), "w") as file:
                pass
    
    pairs = [[tf,zp] for tf in tfs for zp in zps if tf[:-4] in zp]
    
    #make saveLocation if doesn"t exist:
    makedir(saveLocation)
    
    #make mem_mapped arrays once, to be more cluster friendly
    print("Starting conversion...")
    p = mp.Pool(12)
    iterlst = [(pair[0], pair[1], saveLocation) for pair in pairs]
    bad = p.starmap(basic_convert, iterlst)
    p.terminate()
    print ("Completed!\n\nBad list: {}".format(bad)) 
    
    #check all and make dictionary of points (nx3)
    file_points_dct = {}
    print("Checking all files have valid input and anns...")
    for a in listdirfull(saveLocation, "npy"):
        arr = load_np(a)
        sh = np.nonzero(arr[0])[0].shape[0]
        pnts = np.nonzero(arr[1])
        shh = pnts[0].shape[0]
        if sh==0 or shh==0:
            print ("File: {} \n  input images nonzeros=={}, annotation nonzeros=={}".format(a,sh,shh))
            if remove_bad:
                os.remove(a)
                print ("removing")
        else:
            file_points_dct[os.path.basename(a)] = list(zip(*pnts))
            
    #save out points
    save_dictionary(os.path.join(os.path.dirname(saveLocation), "points_dictionary.p"), file_points_dct)
    print("Saved dictionary in {}".format(saveLocation))
        
    return
    
def basic_convert(tf, zp, saveLocation):
    """
    """
    if np.all((os.path.exists(tf), os.path.exists(zp))):
        generate_mem_mapped_array_for_net_training(impth=tf, roipth=zp, 
                                                   dst=os.path.join(saveLocation, os.path.basename(tf)[:-4]+".npy"), 
                                                   verbose = True)     
    else:
        sys.stdout.write("\n^^^^^^^^^^^^^^^^^SKIPPING: Paired files not found for: {} & {}^^^^^^^^^^^^^^^^^^^^\n".format(tf, zp))
        return (tf, zp)
    return
    
def generate_mem_mapped_array_for_net_training(impth, roipth, dst, verbose=True):
    """Function for generating a memory mapped array given path to imagestack and roi zip file
    
    To load array after use:
        np.lib.format.open_memmap(pth_to_array, dtype = "uint16", mode = "r")
    
    Inputs
    ------------------
    impth: path to image stack (assumes tiffstack)
    roipth: path to roi file
    dst: path to save mem mapped array
    annotation_type = 
                    "centers": use if centers of cells were marked using FIJI"s point tool
                
    
    Returns
    -----------------
    array: 
        path to generated memory mapped image volume of shape (2, z, y, x)
        array[0,...] = image uint16
        array[1,...] = rois as 255 pixel values in uint16 array
    
    """
    
    #ensure proper inputs
    assert all((impth[-4:]==".tif", np.any((roipth[-4:]==".zip", roipth[-4:]==".roi")), dst[-4:]==".npy")), "impth, roipth, and dst paths must be strings ending .tif, .zip, and .npy respectively"
    
    #load image file
    if verbose: sys.stdout.write("\n\nLoading tiffstack..."); sys.stdout.flush()
    imstack = tifffile.imread(impth)
    if verbose: sys.stdout.write("done"); sys.stdout.flush()
    
    #dims
    z,y,x = imstack.shape
        
    #init mem mapped array
    arr = np.lib.format.open_memmap(dst, dtype = "uint16", mode = "w+", shape = (2,z,y,x))
    if verbose: sys.stdout.write("\nInitialized memory mapped array:\n    {}".format(dst)); sys.stdout.flush()

    #fill array:
    arr[0,...] = imstack
    
    #load rois
    if verbose: sys.stdout.write("\nLoading rois..."); sys.stdout.flush()
    if ".zip" in roipth:
        import zipfile
        with zipfile.ZipFile(roipth) as zf:
            rois = zf.namelist()
        if verbose: sys.stdout.write("done"); sys.stdout.flush()
        #format ZYX, and remove any rois missaved
        rois_formatted = list(zip(*[map(int, xx.replace(".roi","").split("-")[0:3]) for xx in rois if len(xx.split("-"))==3]))
    else:
        from tools.conv_net.input.read_roi import read_roi
        with open(roipth, "rb") as fl:
            rois = read_roi(fl)
        rois_formatted = [tuple(xx) for xx in rois]
    
    if len(rois_formatted)==0:
        print ("*************Error {}- likely ROIS were mis-saved. Trying to fix, this should be checked.".format(os.path.basename(impth)))
        rois_formatted = list(zip(*[map(int, xx.replace(".roi","").split("-")[0:3]) for xx in rois if len(xx.split("-"))==4]))
    #populate arr; (NOTE: ImageJ has one-based numerics FOR Z but 0 for YX vs np w zero-based numerics for ZYX)
    else:
        if verbose: sys.stdout.write("\nPopulating ROIS..."); sys.stdout.flush()
        arr[1,[xx-1 for xx in rois_formatted[0]], rois_formatted[1], rois_formatted[2]] = 255
        arr.flush()        
    
    if verbose: sys.stdout.write("done.\n\n***Memmapped array generated successfully***\n\n"); sys.stdout.flush()
    
    return arr

#%%
    
if __name__ == "__main__":
    #convert first
    inputFolder = "/jukebox/LightSheetData/rat-brody/processed/201910_tracing/training/raw_data"
    saveLocation = "/jukebox/LightSheetData/rat-brody/processed/201910_tracing/training/arrays"; makedir(saveLocation)
    otsufld = "/jukebox/LightSheetData/rat-brody/processed/201910_tracing/training/otsu"; makedir(otsufld)  
    size = (5,10,10)    
    otsu_factor = 0.8
    
    #convert
    convert_input(inputFolder, saveLocation, remove_bad=False)
    
    #check all
    for a in listdirfull(saveLocation, "npy"):
        sh = np.nonzero(load_np(a))[0].shape[0]
        if sh==0: print(a,sh)
#%%                
    #otsu_par
    otsu_par(saveLocation, otsufld, size, otsu_factor)   
