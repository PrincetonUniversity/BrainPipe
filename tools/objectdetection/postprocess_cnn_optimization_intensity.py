#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:54:32 2018

@author: tpisano
"""

from tools.utils.io import makedir, chunkit, load_np, load_dictionary, save_kwargs, listdirfull, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer as dd, pth_update
import numpy as np, multiprocessing as mp,os, pandas as pd, gc,sys
from tools.conv_net.functions.bipartite import pairwise_distance_metrics_multiple_cutoffs
from tools.conv_net.apply_cnn.evaluate_cnn import compute_p_r_f1
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from itertools import product
from tools.conv_net.functions.sphericity import filter_sphericity
from tools.conv_net.functions.size_filter import size_filter
from skimage.external import tifffile
from tools.objectdetection.run_cnn import apply_cnn_to_folder
from tools.objectdetection.postprocess_cnn import load_tiff_folder

if __name__ == '__main__':
    from tools.objectdetection.postprocess_cnn_optimization_intensity import find_3d_peaks_sweep
           
    print sys.argv
    cores = int(sys.argv[1])
    core = int(sys.argv[2])
    
    mem_map_folder = os.path.join(dd(), 'wang/pisano/conv_net/annotations/better_res/memmap_arrays_for_optimization')#os.path.join(dd(), 'wang/pisano/conv_net/annotations/better_res/memmap_arrays')
    cnn_prediction_folder=os.path.join(dd(), 'wang/pisano/conv_net/postprocessing_optimization/memmap_arrays_cnn')
    '''Only needed to do once
    #^^need to make sure the cnn output exists there. See cnn_performance_quantifcatiyon
    for ii,pth in enumerate([xx for xx in os.listdir(mem_map_folder) if 'false' not in xx and 'label' not in xx]):
        #load, save out, and run cnn, and filter size
        arr = load_np(os.path.join(mem_map_folder, pth))[0]
        dst0 = os.path.join(cnn_prediction_folder, pth); makedir(dst0)
        [tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), im) for i,im in enumerate(arr)]
        apply_cnn_to_folder(src=dst0, dst=dst0+'_cnn_output')
    '''
   
    raws = []; gts = []; cnns=[]
    for fl in listdirfull(mem_map_folder):
        raws.append(load_np(fl)[0])
        gts.append(load_np(fl)[1])
        cnns.append([load_tiff_folder(xx, threshold = 1) for xx in listdirfull(cnn_prediction_folder, keyword = 'cnn_output') if os.path.basename(fl) in xx][0])
        
    raw = np.stack(raws, axis=0)
    ground_truth = np.stack(gts, axis=0)
    cnn = np.stack(cnns, axis=0)
    dst=os.path.join(dd(), 'wang/pisano/conv_net/postprocessing_optimization/param_sweep_intensity'); makedir(dst)
    kwargs = {'dst':dst, 'cnn':cnn, 'raw':raw, 'core':core, 'cores':cores, 'ground_truth':ground_truth, 'return_kwargs':True, 'xyz_pixel_distance_ratio':(1.63, 1.63, 7.5)}
    find_3d_peaks_sweep(kwargs)
    
if False:
    #load
    df = pd.concat([pd.read_csv(xx) for xx in listdirfull(os.path.join(dd(), 'wang/pisano/conv_net/postprocessing_optimization/param_sweep_intensity'))])        
    
    
#%%


def find_3d_peaks_sweep(kwargs):
    '''passing kwargs without ** for parallelization, this is used for spock to prevent memory issues, see find_3d_peaks for full description

    Inputs
    ----------
    cnn = kwargs['cnn'], 4d arrays
    raw = kwargs['raw'], 4d arrays
    groundtruth - assumes this is a numpy array with nonzeros at pnts
    zyx_patch_delta_border - (10,10,10), used if there is a size difference between the groundtruth array and cnnpatches. This is the border size (i.e. full difference / 2)
    dst
    core = int
    cores = int
    
    iterdct = {'min_distance': [13,17,21],
            'sphericity': [0,.15,.2,.25,.3, .4, .5,],
            'sphericity_dims': [2,3],
            'size_filter_lower': range(160,210,10), #[0,2,8,16,25],
            'size_filter_upper': range(3500,5000,250)}#[1000,2000,10000]} 
    
    xyz_pixel_distance_ratio (1.63, 1.63, 7.5) <- used to scale for cutoff distances ONLY not filtering. Scales both groundtruth and discovered centers
    Saves out as csv

    '''
    #inputs
    assert np.all((kwargs['raw'].ndim == 4, kwargs['cnn'].ndim == 4, kwargs['ground_truth'].ndim == 4)), '4d arrs expected for both src and ground truth'
    cnn = np.copy(load_np(kwargs['cnn'])) if type(kwargs['cnn'])==str else np.copy(kwargs['cnn'])
    raw = np.copy(load_np(kwargs['raw'])) if type(kwargs['raw'])==str else np.copy(kwargs['raw'])
    if cnn.ndim == 4: src = np.stack((raw, cnn), 1)
    if cnn.ndim == 3: src = np.stack((raw, cnn), 0)    
    iterdct = {#'min_distance': [10, 20, 30],
            'sphericity': [0,.05,.1,.15,.2,.25,.3,],
            'sphericity_dims': [2], #[2,3]
            'size_filter_lower': range(90,230,10), 
            'size_filter_upper': range(3750,4750,250),
            'intensity_lower': range(0,8000, 500)}

     
    iterlst = [dict(zip(iterdct.keys(),xx)) for xx in product(*iterdct.values())]
    print ('{} iterations'.format(len(iterlst)))
    l, h = chunkit(kwargs['core'], kwargs['cores'], iterlst)
    
    
    kwargs['ground_truth'] = np.asarray(np.nonzero(kwargs['ground_truth'])).T

    #optional
    if 'xyz_pixel_distance_ratio' in kwargs:
        xs,ys,zs = kwargs['xyz_pixel_distance_ratio']
        kwargs['ground_truth'] = np.asarray((kwargs['ground_truth'][:,0], kwargs['ground_truth'][:,1]*zs, kwargs['ground_truth'][:,2]*ys, kwargs['ground_truth'][:,3]*xs)).T.astype('int64')
        
    
    for i in range(l,h):
      csrc = np.copy(src)
      dct = iterlst[i]
      sp = dct['sphericity']; spd = dct['sphericity_dims']; sfl = dct['size_filter_lower']; sfu = dct['size_filter_upper']; inten = dct['intensity_lower']
      if not os.path.exists(os.path.join(kwargs['dst'], 'sp{}_spd{}_sfl{}_sfu{}_inten{}.csv'.format(sp,spd,sfl, sfu, inten))):
        if csrc.ndim == 4:
            centers = np.asarray(find_3d_peaks_helper(csrc, sp, spd, sfl, sfu, inten))
        else:
            centers = np.asarray([zz for yy in [[(i, xx[0], xx[1], xx[2]) for xx in find_3d_peaks_helper(s, sp, spd, sfl, sfu, inten)] for i,s in enumerate(src)] for zz in yy])
        
        #optional
        if 'ground_truth' in kwargs:
            if 'xyz_pixel_distance_ratio' in kwargs and centers.size!=0: centers = np.asarray((centers[:,0], centers[:,1]*zs, centers[:,2]*ys, centers[:,3]*xs)).T.astype('int64')
            cutoffs = kwargs['cutoffs'] if 'cutoffs' in kwargs else [0.1, 1.0, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125]
            df = pd.DataFrame(data=None, columns = ['cutoff', 'sphericity', 'sphericity_dims', 'size_filter_lower', 'size_filter_upper', 'intensity_lower', 'tp', 'fp', 'fn', 'p','r','f1'])
            metrics = pairwise_distance_metrics_multiple_cutoffs(kwargs['ground_truth'], centers, verbose=False, return_paired=False, cutoffs=cutoffs)
            for i, cutoff in enumerate(cutoffs):
                tp,fp,fn = metrics[i]
                p,r,f1 = compute_p_r_f1(tp=tp,fp=fp,fn=fn)
                df.loc[len(df)] = [cutoff, sp, spd, sfl, sfu, inten, tp,fp, fn, p, r, f1]
                del tp,fp,fn,p,r,f1
            del metrics
    
        makedir(kwargs['dst'])
        df.to_csv(os.path.join(kwargs['dst'], 'sp{}_spd{}_sfl{}_sfu{}_inten{}.csv'.format(sp,spd,sfl, sfu, inten)))
        del dct, csrc, centers, df; gc.collect()
    print ('Completed {}-{}'.format(l,h))
    return


def find_3d_peaks_helper(src, sp, spd, sfl, sfu, inten):
    '''Helper function - currently only filtering on sphericity for the first labeling (before "declustering").
    '''
    raw = src[0]; cnn = src[1]
    #find connected components and local distances
    labels = ndi.measurements.label(cnn)
    #filter to remove too small or large connected components
    labels = size_filter(labels, lower=sfl, upper=sfu)
    #filter each label for sphericity <-custommade function
    labels = filter_sphericity(labels, cutoff = sp, dims=spd) if sp > 0 else labels
    #mask labels onto raw
    raw[labels[0]==0] = 0
    #filter by intensity
    raw = filter_by_intensity(labels, raw, inten); del labels
    #find distance image
    #distance = ndi.distance_transform_edt(s); del s
    #now find peak local max of each connected component - done to possibly separate close cells, dtype is really important here
    #local_maxi = peak_local_max(distance, labels=labels[0].astype('float32'),  min_distance=min_distance, exclude_border=False, indices=False).astype('float32'); del labels
    #finally find centers
    #lbl = ndi.label(local_maxi); del local_maxi
    lbl = ndi.label(raw)
    return ndi.measurements.center_of_mass(raw, lbl[0], range(1, lbl[1]+1))
    #return ndi.measurements.center_of_mass(distance, lbl[0], range(1, lbl[1]+1))

def filter_by_intensity(labels, raw, inten):
    '''finds maximum in connected component, if lower than inten removes
    '''
    for val in np.unique(labels[0])[1:]:
        pxls = np.asarray(np.where(labels[0]==val)).T
        if np.max([raw[xx[0],xx[1],xx[2]] for xx in pxls]) <= inten: 
            for xx in pxls:
                raw[xx[0],xx[1],xx[2]]=0
    return raw
    
    
    
    
    
    