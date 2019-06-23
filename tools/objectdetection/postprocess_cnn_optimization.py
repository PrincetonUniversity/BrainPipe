#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:09:19 2018

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
    from tools.objectdetection.postprocess_cnn_optimization import find_3d_peaks_sweep
           
    print sys.argv
    cores = int(sys.argv[1])
    core = int(sys.argv[2])
    
    mem_map_folder = os.path.join(dd(), 'wang/pisano/conv_net/annotations/better_res/memmap_arrays')#os.path.join(dd(), 'wang/pisano/conv_net/annotations/better_res/memmap_arrays_for_optimization')
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
   
    gts = []; srcs=[]
    for fl in listdirfull(mem_map_folder):
        gts.append(load_np(fl)[1])
        srcs.append([load_tiff_folder(xx, threshold = 1) for xx in listdirfull(cnn_prediction_folder, keyword = 'cnn_output') if os.path.basename(fl) in xx][0])
        
    ground_truth = np.stack(gts, axis=0)
    src = np.stack(srcs, axis=0)
    dst=os.path.join(dd(), 'wang/pisano/conv_net/postprocessing_optimization/param_sweep_3'); makedir(dst)
    kwargs = {'dst':dst, 'src':src, 'core':core, 'cores':cores, 'ground_truth':ground_truth, 'return_kwargs':True, 'xyz_pixel_distance_ratio':(1.63, 1.63, 7.5)}
    find_3d_peaks_sweep(kwargs)
    
    
if __name__ == '__local__':
    
    dst='/home/wanglab/wang/pisano/conv_net/postprocessing_optimization'; makedir(dst)
    
    ####Only need to be done once!########
    fls = listdirfull('/home/wanglab/wang/pisano/conv_net/annotations/better_res/memmap_arrays_for_optimization', keyword = '.npy')
    for fl in fls:
        src = load_np(fl)[0]
        dst0 = os.path.join(dst, os.path.basename(fl)); makedir(dst0)
        for i in range(len(src)):
            tifffile.imsave(os.path.join(dst0, '{}.tif'.format(str(i).zfill(4))), src[i])
        dst1 = os.path.join(dst, os.path.basename(fl)+'_cnn_output'); makedir(dst1)
        apply_cnn_to_folder(dst0, dst1, matlab_file=False, cnn_weights=False)
    #####################################
    
    gts = []; srcs=[]
    for fl in listdirfull('/home/wanglab/wang/pisano/conv_net/annotations/better_res/memmap_arrays_for_optimization'):
        gts.append(load_np(fl)[1])
        srcs.append([load_tiff_folder(xx, threshold = 1) for xx in listdirfull(dst, keyword = 'cnn_output') if os.path.basename(fl) in xx][0])
        
    ground_truth = np.stack(gts, axis=0)
    src = np.stack(srcs, axis=0)
    cores=12
    dst='/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/param_sweep_2'; makedir(dst)
    iterlst = [{'dst':dst, 'src':src, 'core':core, 'cores':cores, 'ground_truth':ground_truth, 'return_kwargs':True, 'xyz_pixel_distance_ratio':(1.63, 1.63, 7.5)} for core in range(cores)]
    p = mp.Pool(cores)
    output = p.map(find_3d_peaks_sweep, iterlst)
    p.terminate()
    
    #collect
    adf = pd.read_csv('/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/size_filter/performance.csv')
    df = pd.concat([pd.read_csv(xx) for xx in listdirfull('/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/param_sweep_2')])
    df = pd.concat([pd.read_csv(xx) for xx in listdirfull('/home/wanglab/wang/pisano/conv_net/postprocessing_optimization/param_sweep_3')])
    

#%%


def find_3d_peaks_sweep(kwargs):
    '''passing kwargs without ** for parallelization, this is used for spock to prevent memory issues, see find_3d_peaks for full description

    Inputs
    ----------
    srcs = kwargs['src'], 4d arrays
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
    assert np.all((kwargs['src'].ndim == 4, kwargs['ground_truth'].ndim == 4)), '4d arrs expected for both src and ground truth'
    src = np.copy(load_np(kwargs['src'])) if type(kwargs['src'])==str else np.copy(kwargs['src'])
        
    iterdct = {'min_distance': [17,21,25,27,31,35,39,43],
            'sphericity': [.25,.3,.35],
            'sphericity_dims': [2], #[2,3]
            'size_filter_lower': range(180,225,5), 
            'size_filter_upper': range(3500,5000,250)}

     
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
      min_distance = dct['min_distance']; sp = dct['sphericity']; spd = dct['sphericity_dims']; sfl = dct['size_filter_lower']; sfu = dct['size_filter_upper']
      if not os.path.exists(os.path.join(kwargs['dst'], 'md{}_sp{}_spd{}_sfl{}_sfu{}.csv'.format(min_distance,sp,spd,sfl, sfu))):
        if csrc.shape == 3:
            centers = np.asarray(find_3d_peaks_helper(csrc, min_distance, sp, spd, sfl, sfu))
        else:
            centers = np.asarray([zz for yy in [[(i, xx[0], xx[1], xx[2]) for xx in find_3d_peaks_helper(s, min_distance, sp, spd, sfl, sfu)] for i,s in enumerate(src)] for zz in yy])
        
        #optional
        if 'ground_truth' in kwargs:
            if 'xyz_pixel_distance_ratio' in kwargs and centers.size!=0: centers = np.asarray((centers[:,0], centers[:,1]*zs, centers[:,2]*ys, centers[:,3]*xs)).T.astype('int64')
            cutoffs = kwargs['cutoffs'] if 'cutoffs' in kwargs else [0.1, 1.0, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125]
            df = pd.DataFrame(data=None, columns = ['cutoff', 'min_distance',  'sphericity', 'sphericity_dims', 'size_filter_lower', 'size_filter_upper', 'tp', 'fp', 'fn', 'p','r','f1'])
            metrics = pairwise_distance_metrics_multiple_cutoffs(kwargs['ground_truth'], centers, verbose=False, return_paired=False, cutoffs=cutoffs)
            for i, cutoff in enumerate(cutoffs):
                tp,fp,fn = metrics[i]
                p,r,f1 = compute_p_r_f1(tp=tp,fp=fp,fn=fn)
                df.loc[len(df)] = [cutoff, min_distance, sp, spd, sfl, sfu, tp,fp, fn, p, r, f1]
                del tp,fp,fn,p,r,f1
            del metrics
    
        makedir(kwargs['dst'])
        df.to_csv(os.path.join(kwargs['dst'], 'md{}_sp{}_spd{}_sfl{}_sfu{}.csv'.format(min_distance,sp,spd,sfl, sfu)))
        del dct, csrc, centers, df; gc.collect()
    print ('Completed {}-{}'.format(l,h))
    return


def find_3d_peaks_helper(s, min_distance, sp, spd, sfl, sfu):
    '''Helper function - currently only filtering on sphericity for the first labeling (before "declustering").
    '''
    #find connected components and local distances
    labels = ndi.measurements.label(s)
    #filter to remove too small or large connected components
    labels = size_filter(labels, lower=sfl, upper=sfu)
    #filter each label for sphericity <-custommade function
    labels = filter_sphericity(labels, cutoff = sp, dims=spd) if sp > 0 else labels
    #find distance image
    distance = ndi.distance_transform_edt(s); del s
    #now find peak local max of each connected component - done to possibly separate close cells, dtype is really important here
    local_maxi = peak_local_max(distance, labels=labels[0].astype('float32'),  min_distance=min_distance, exclude_border=False, indices=False).astype('float32'); del labels
    #finally find centers
    lbl = ndi.label(local_maxi); del local_maxi
    #kwargs['labels'] = lbl
    return ndi.measurements.center_of_mass(distance, lbl[0], range(1, lbl[1]+1))