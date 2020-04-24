#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:42:10 2018

@author: wanglab

most functions modified from @tpisano's lightsheet repository
"""

import os, h5py, cv2, zipfile, sys, matplotlib.pyplot as plt, numpy as np, pickle, time, pandas as pd, ast
from subprocess import check_output
from skimage.external import tifffile
from utils.postprocessing.cell_stats import consolidate_cell_measures
from scipy.spatial import distance

def pairwise_distance_metrics_given_cdists(ground_truth, predicted, y, cutoff=10, verbose=True):
    """
    Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
    y: matrix of distances between all elements of ground truth and predicted
                        
    Returns:
    -------
    paired: list of [ground_truth"s index (from input list), predicted"s index (from input list), distance]
    tp,fp,fn: statistics on true positives, false positives, and false negatives.
    """   
    
    start=time.time()
    if verbose: print("\nCalculating pairwise distances at cutoff {}...".format(cutoff))
    st = time.time()
    #only keep those distances that are below the cutoff!
    truth_indices,pred_indices = np.where(y <= cutoff)
    dists = zip(y[truth_indices,pred_indices],truth_indices,pred_indices)
    if verbose: print("  {} seconds for calculating and collecting distances".format(np.round(time.time() - st, decimals=3)))

    #sort by smallest dist
    if verbose: print("Sorting..."); st = time.time()
    dists=sorted(dists, key=lambda x: x[0])
    if verbose: sys.stdout.write("  Sorted in {} seconds.".format(np.round(time.time() - st, decimals=3))); sys.stdout.flush()

    used_truth = set()
    used_pred = set()
    paired = []
    for (i,dist) in enumerate(dists):
        d = dist[0]
        if d > cutoff:
            #we have reached distances beyond the cutoff
            if verbose: print("Reached cutoff distance, so far {} paired".format(len(paired)))
            break
        truth_idx = dist[1]
        pred_idx = dist[2]
        if truth_idx not in used_truth and pred_idx not in used_pred:
            paired.append((truth_idx,pred_idx,d))
            used_truth.add(truth_idx)
            used_pred.add(pred_idx)
        if len(used_truth) == len(ground_truth) or len(used_pred) == len(predicted):
            # we have used up all the entries from the shorter list
            break

    tp = len(paired)
    fn = len(ground_truth) - len(paired)
    fp = len(predicted) - len(paired)
    if verbose: print("TP: {}, FP: {}, FN: {}".format(tp,fp,fn))
    # print(paired)
  
    if verbose: 
        plt.hist([xx[2] for xx in paired] , bins = np.max((int(len(paired)/500), 10)))
        plt.title("Histogram of distances - pixel or microns")
    
    if verbose: print("Finished in {} seconds\n".format(np.round(time.time() - start,decimals = 3)))
    
    return paired,tp,fp,fn


def pairwise_distance_metrics(ground_truth, predicted, cutoff=10, verbose=True):
    """
    Function to calculate the pairwise distances between two lists of zyx points.
    
    Inputs:
    -------
    ground_truth, predicted: each iterable consisting of ndimensional coordinates.
                        
    Returns:
    -------
    paired: list of [ground_truth"s index (from input list), predicted"s index (from input list), distance]
    tp,fp,fn: statistics on true positives, false positives, and false negatives.
    """   
    
    if verbose: print("\nCalculating pairwise distances...")
    y = distance.cdist(ground_truth, predicted, metric="euclidean")
    return pairwise_distance_metrics_given_cdists(ground_truth,predicted,y,cutoff,verbose)


def swap_cols(arr, frm, to):
    """
    helper function used to swap np array columns if orientation changes have been made pre-registration
    """
    try:
        arr[:, [frm, to]]=arr[:, [to, frm]]
    except:
        print ("array is likely empty - and so need to adjust thresholding")
    return arr

def read_roi(fileobj):
    """
    points = read_roi(fileobj)

    Read ImageJ"s ROI format. Points are returned in a nx2 array. Each row
    is in [row, column] -- that is, (y,x) -- order.
    Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
            Tim D. Smith <git@tim-smith.us>, 2015
    License: MIT
    
    """
    # This is based on:
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html

    SUB_PIXEL_RESOLUTION = 128

    class RoiType:
        POLYGON = 0
        RECT = 1
        OVAL = 2
        LINE = 3
        FREELINE = 4
        POLYLINE = 5
        NOROI = 6
        FREEHAND = 7
        TRACED = 8
        ANGLE = 9
        POINT = 10

    def get8():
        s = fileobj.read(1)
        if not s:
            raise IOError("readroi: Unexpected EOF")
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != b"Iout":
        raise ValueError("Magic number not found")

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()

    if roi_type not in [RoiType.FREEHAND, RoiType.POLYGON, RoiType.RECT, RoiType.POINT]:
        raise NotImplementedError("roireader: ROI type %s not supported" % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()
    x1 = getfloat()
    y1 = getfloat()
    x2 = getfloat()
    y2 = getfloat()
    subtype = get16()
    if subtype != 0:
        raise NotImplementedError("roireader: ROI subtype %s not supported (!= 0)" % subtype)
    options = get16()

    if roi_type == RoiType.RECT:
        if options & SUB_PIXEL_RESOLUTION:
            return np.array(
                [[y1, x1], [y1, x1+x2], [y1+y2, x1+x2], [y1+y2, x1]],
                dtype=np.float32)
        else:
            return np.array(
                [[top, left], [top, right], [bottom, right], [bottom, left]],
                dtype=np.int16)

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
        fileobj.seek(4*n_coordinates, 1)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)

    points[:, 1] = [getc() for i in range(n_coordinates)]
    points[:, 0] = [getc() for i in range(n_coordinates)]

    if options & SUB_PIXEL_RESOLUTION == 0:
        points[:, 1] += left
        points[:, 0] += top

    return points


def read_roi_zip(fname, include_roi_name=False, verbose=True):
    """
    Wrapper for reading zip files generated from ImageJ (FIJI)
    
    include_roi_name (optional) 
        if true: returns list of (roi_name, contour)
        roi_name=z,y,x
        useful for extracting z (NOTE: ImageJ has one-based numerics vs Python w zero-based numerics)
    """
        
    try:
        if not include_roi_name:
            with zipfile.ZipFile(fname) as zf:
                return [read_roi(zf.open(n)) for n in zf.namelist()]
                                                    
        if include_roi_name:
            with zipfile.ZipFile(fname) as zf:
                return [(n, read_roi(zf.open(n))) for n in zf.namelist()]
    
    #hack to try and keep 
    except ValueError:
        lst = []
        with zipfile.ZipFile(fname) as zf:
            for n in zf.namelist():
                if len( n[:-4].split("-")) == 3:
                     lst.append(n)
        return lst
    
def load_dictionary(pth):
    """
    simple function to load dictionary given a pth
    """
    kwargs = {};
    with open(pth, 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs
    
def save_dictionary(dst, dct):
    
    if dst[-2:]!=".p": dst=dst+".p"
    
    with open(dst, "wb") as fl:    
        pickle.dump(dct, fl, protocol=pickle.HIGHEST_PROTOCOL)
    return

def makedir(path):
    """ simple function to make directory if path does not exists """
    if os.path.exists(path) == False:
        os.mkdir(path)
    return

def listdirfull(x, keyword=False):

    if not keyword:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx]
    else:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx and keyword in xx]

def load_memmap_arr(pth, mode="r", dtype = "uint16", shape = False):
    """
    Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | "r"  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | "r+" | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | "w+" | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | "c"  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    """
    if shape:
        assert mode =="w+", "Do not pass a shape input into this function unless initializing a new array"
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

def load_np(src, mode="r"):
    """
    Function to handle .npy and .npyz files. Assumes only one k,v pair in npz file
    """
    if str(type(src)) == "<type numpy.ndarray>" or str(type(src)) == "<class numpy.core.memmap.memmap>":
        return src
    elif src[-4:]==".npz":
        fl = np.load(src)
        #unpack ASSUMES ONLY SINGLE FILE
        arr = [fl[xx] for xx in fl.keys()][0]
        return arr
    elif src[-4:]==".npy":
        try:
            arr = load_memmap_arr(src, mode)
        except:
            arr = np.load(src)
        return arr
    
#function to run
def sp_call(call):
    """ command line call function """ 
    print(check_output(call, shell=True)) 
    return


def make_inference_output_folder(pth):
    """ needed to start inference correctly so chunks aren"t missing from output folder """
    
    if not os.path.exists(os.path.join(pth, "output_chnks")): os.mkdir(os.path.join(pth, "output_chnks"))
    print("output folder made for :\n {}".format(pth))
    
    return

def consolidate_cell_measures_bulk(pth):
    
    fls = [xx for xx in os.listdir(pth) if "reconstructed_array.npy" in os.listdir(os.path.join(pth, xx))]
    
    for fl in fls:
        src = os.path.join(os.path.join(pth, fl), "cnn_param_dict.csv")
        params = csv_to_dict(src)
        consolidate_cell_measures(**params)
    
    return
    
def resize(pth, dst, resizef = 6):
    """ 
    resize function using cv2
    inputs:
        pth = 3d tif stack or memmap array
        dst = folder to save each z plane
    """
    #make sure dst exists
    if not os.path.exists(dst): os.mkdir(dst)
    
    #read file
    if pth[-4:] == ".tif": img = tifffile.imread(pth)
    elif pth[-4:] == ".npy": img = np.lib.format.open_memmap(pth, dtype = "float32", mode = "r")
    
    z,y,x = img.shape
    
    for i in range(z):
        #make the factors
        xr = img[i].shape[1] / resizef; yr = img[i].shape[0] / resizef
        im = cv2.resize(img[i], (xr, yr), interpolation=cv2.INTER_LINEAR)
        tifffile.imsave(os.path.join(dst, "zpln{}.tif".format(str(i).zfill(12))), im.astype("float32"), compress=1)
    
    return dst

def resize_stack(pth, dst):
    
    """
    runs with resize
    inputs:
        pth = folder with resized tifs
        dst = folder
    """
    #make sure dst exists
    if not os.path.exists(dst): os.mkdir(dst)
    
    #get all tifs
    fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx and ".tif" in xx]; fls.sort()
    y,x = tifffile.imread(fls[0]).shape
    dims = (len(fls),y,x)
    stack = np.zeros(dims)
    
    for i in range(len(fls)):
        stack[i] = tifffile.imread(fls[i])
    
    #save stack
    tifffile.imsave(os.path.join(dst, "resized_stack.tif"), stack.astype("float32"))
    
    return os.path.join(dst, "resized_stack.tif")
    
def check_dim(pth):
    """ 
    find all dimensions of imgs in the direccvtory 
    usefull to check training inputs before setting window size
    i.e. window size should not be larger than input dimensions 
    e.g. pth = "/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages"
    only h5 files
    """
    for i, fn in enumerate(os.listdir(pth)):
        f = h5py.File(os.path.join(pth,fn))
        d = f["/main"].value
        f.close()
        print(fn, d.shape, np.nonzero(d)[0].shape)

def sample_reconstructed_array(pth, zstart, zend):
    """ check to make sure reconstruction worked
    pth = path to cnn output folder (probably in scratch) that has the reconstructed array
    """

    flds = os.listdir(pth)
    if "reconstructed_array.npy" in flds: 
        #read memory mapped array
        chunk = np.lib.format.open_memmap(os.path.join(pth, "reconstructed_array.npy"), dtype = "float32", mode = "r")
        print(chunk.shape)
        
        #save tif
        tifffile.imsave(os.path.join(pth, "sample.tif"), chunk[zstart:zend, :, :])
        
        print("chunk saved as: {}".format(os.path.join(pth, "sample.tif")))
    
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

