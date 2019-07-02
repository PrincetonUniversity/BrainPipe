#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:55:37 2019

@author: wanglab
"""

import os, numpy as np, sys, multiprocessing as mp
from skimage.external import tifffile
from skimage import filters
import pickle

def listdirfull(x, keyword=False):
    '''might need to modify based on server...i.e. if automatically saving a file called 'thumbs'
    '''
    if not keyword:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx]
    else:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx and keyword in xx]
    
    
def otsu_par(saveLocation, otsufld, guassian_sigma, otsu_factor):
   
    #otsu
    p = mp.Pool(12)
    iterlst = [(otsufld, inn, guassian_sigma, otsu_factor) for inn in listdirfull(saveLocation, "npy")]
    p.starmap(otsu_helper, iterlst)
    p.terminate()
    return
            
def otsu_helper(otsufld, inn, guassian_sigma, otsu_factor):
    
    #load
    arr = np.load(inn)
    raw = np.copy(arr[0])
    lbl = np.copy(arr[1])
    
    #save input
    tifffile.imsave(os.path.join(otsufld, "{}_img.tif".format(os.path.basename(inn)[:-4])), raw.astype("float32"))
    
    #save input
    otsu = otsu_dilate(raw, lbl, sigma = guassian_sigma, otsu_factor=otsu_factor)
    tifffile.imsave(os.path.join(otsufld, "{}_lbl.tif".format(os.path.basename(inn)[:-4])), otsu.astype("float32"))

    print(inn)
    
    return

def otsu_dilate(arr0, arr1, sigma, otsu_factor=0.8):
    """4d arr
    arr0=raw data
    arr1=points
    size=(z,y,x)
    otsu_factor - scaling of the otsu value, >1 is less stringent, <1 remove more pixels
    """
    vol = filters.gaussian(arr1, sigma = sigma)
    v = filters.threshold_otsu(vol)/float(otsu_factor)
    vol[vol < v] = 0
    vol[vol >= v] = 1
    
    return vol.astype("uint16")

def convert_input(inputFolder, saveLocation, remove_bad=True):
    """Function for converting data from imageJ ROIs + data to mem_mapped arrays for preprocessing + batch generation to pass to cnn
    """
    #get pairs
    tfs = listdirfull(inputFolder,keyword=".tif")
    zps = [xx for xx in listdirfull(inputFolder) if ".tif" not in xx]
    pairs = [[tf,zp] for tf in tfs for zp in zps if tf[:-4] in zp]

    #make mem_mapped arrays once, to be more cluster friendly
    import multiprocessing as mp
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
        arr = np.load(a)
        sh = np.nonzero(arr[0])[0].shape[0]
        pnts = np.nonzero(arr[1])
        shh = pnts[0].shape[0]
        if sh==0 or shh==0:
            print ("File: {} \n  input images nonzeros=={}, annotation nonzeros=={}".format(a,sh,shh))
            if remove_bad:
                os.remove(a)
                print ("removing")
        else:
            file_points_dct[os.path.basename(a)] = zip(*pnts)
            
    #save out points
    dst = os.path.join(os.path.dirname(saveLocation), "filename_points_dictionary.p")
    
    with open(dst, 'wb') as fl:    
        pickle.dump(file_points_dct, fl, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Saved dictionary as {}".format(os.path.join(os.path.dirname(saveLocation), "filename_points_dictionary.p")))
        
    return
    
def basic_convert(tf, zp, saveLocation):
    """
    """
    if np.all((os.path.exists(tf), os.path.exists(zp))):
        generate_mem_mapped_array_for_net_training(impth=tf, roipth=zp, 
                                                   dst=os.path.join(saveLocation, os.path.basename(tf)[:-4]+".npy"), verbose = True)     
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
        rois_formatted = zip(*[map(int, xx.replace(".roi","").split("-")[0:3]) for xx in rois if len(xx.split("-"))==4])
    #populate arr; (NOTE: ImageJ has one-based numerics FOR Z but 0 for YX vs np w zero-based numerics for ZYX)
    if verbose: sys.stdout.write("\nPopulating ROIS..."); sys.stdout.flush()
    arr[1,[xx-1 for xx in rois_formatted[0]], rois_formatted[1], rois_formatted[2]] = 255

    
    arr.flush()        
    if verbose: sys.stdout.write("done.\n\n***Memmapped array generated successfully***\n\n"); sys.stdout.flush()
    
    return arr

#%%
    
if __name__ == "__main__":
    
    #convert first
    inputFolder = "/home/wanglab/Documents/cfos_raw_inputs/"
    saveLocation = "/home/wanglab/Documents/cfos_inputs/memmap"
    if not os.path.exists(saveLocation): os.mkdir(saveLocation) #test folder that contains memory mapped arrays will img + lbl points
    thresfld = "/home/wanglab/Documents/cfos_inputs/otsu_and_guassian"
    if not os.path.exists(thresfld): os.mkdir(thresfld) #output folder
    otsu_factor = 4
    guassian_sigma = 1
    #convert
    convert_input(inputFolder, saveLocation, remove_bad=True)
    
    #check all
    for a in listdirfull(saveLocation, "npy"):
        sh = np.nonzero(np.load(a))[0].shape[0]
        if sh==0: print(a,sh)
#%%        
    #otsu_par
    otsu_par(saveLocation, thresfld, guassian_sigma, otsu_factor)  