#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:23:54 2017

@author: wanglab
"""
from __future__ import division
from tools.utils import *
from tools.objectdetection import find_cells
import multiprocessing as mp
import numpy as np
import cPickle as pickle
import os, sys, time, warnings, collections
from skimage.external import tifffile
import SimpleITK as sitk
from tools.imageprocessing.preprocessing import makedir, listdirfull, writer, resample_par
from tools.registration.register import make_inverse_transform, point_transform_due_to_resizing, point_transformix, transformed_pnts_to_allen
from tools.objectdetection.three_d_celldetection import detect_cells_in_3d
from tools.registration.transform import identify_structures_w_contours
from tools.objectdetection.injdetect import inj_detect_using_labels
import matplotlib.pyplot as plt
from skimage import exposure
from tools.utils.io import load_kwargs
#%%


def testing_cell_detect_params(image, xyz_scale=(1,1,1), cell_area_thresh=[45,1850], cell_circularity_thresh=0.3, abs_intensity_percentile_thresh=0.2, edge_finding_param=0.99, sigma=2, dilation_kernel_size=2, rng=None, show_all = False, show_final=True):
    '''Function to optimize cell detection parameters.
    _______
    Inputs:
        image = numpy array, filepath to folder of tiffs OR single tiff
        xyz_scale = pixel to micron distance; 4x obj = 1.63, 1.3x = 5
        cell_area_threshold = min and max of allowable cell area
        cell_circularity_threshold = allowable cell circularity; 1 = 'perfect' circle
        abs_intensity_percentile_thresh = upper percentile pixel intensitys to consider as cells; a finicky parameter when set high
        edge_finding_param
        sigma, 
        dilation_kernel_size
        rng = optional; list of ints that should be loaded (might be broken)
        show = loads the image into imagej using Simple ITK
    ________    
    Outputs:
        numpy array of image
    '''    
    ###load image
    if type(image) == np.ndarray:
        im=image

    elif type(image) == str:

        if image[-4:] == '.tif':
            
            if rng != None:            
                im = tifffile.imread(image, key=rng)
            
            else:
                im = tifffile.imread(image)
        
        else:
            impths = [os.path.join(image, xx) for xx in os.listdir(image)]
            
            if rng == None:
                rng = impths

            y,x = tifffile.imread(impths[0]).shape            
            im = np.zeros((len(rng), y,x))
            
            for z in range(len(rng)):
                im[z,...] = tifffile.imread(impths[z])
    
    #check bit depth of image and rescale if necessary:
    if im.dtype != 'uint16':
        im = exposure.rescale_intensity(im, in_range=str(im.dtype), out_range='uint16')
                
    ###cell detect
    newim=np.zeros(im.shape)    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        if show_all:
            assert len(newim.shape) == 2, 'Cannot use show_all=True with an image stack. Use this feature with only a single 2D image'
            if len(newim.shape) == 2:
                newim, centers, contours, edge_im = find_cells.cell_detect(im, cell_area_thresh=cell_area_thresh, cell_circularity_thresh=cell_circularity_thresh, abs_intensity_percentile_thresh=abs_intensity_percentile_thresh, xyz_scale=xyz_scale, edge_finding_param=edge_finding_param, sigma=sigma, dilation_kernel_size=dilation_kernel_size, showedgeim = True)         
                print ('{} objects detected in plane'.format(len(contours)))
                
                #make plot:
                plt.ion()
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
                f.suptitle('cell_area_thresh {}, cell_circularity_thresh {}, abs_intensity_percentile_thresh {}, edge_finding_param {}, sigma {}, dilation_kernel_size {}'.format(cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh, edge_finding_param, sigma, dilation_kernel_size))
                ax1.imshow(im)
                ax1.set_title('Original')
                ax2.imshow(edge_im)
                ax2.set_title('Edge Im')
                ax3.imshow(newim)
                ax3.set_title('Overlay')


        elif not show_all:
            if len(newim.shape) == 2:
                tmp = find_cells.cell_detect(im.astype('uint16'), cell_area_thresh=cell_area_thresh, cell_circularity_thresh=cell_circularity_thresh, abs_intensity_percentile_thresh=abs_intensity_percentile_thresh, xyz_scale=xyz_scale, edge_finding_param=edge_finding_param, sigma=sigma, dilation_kernel_size=dilation_kernel_size)         
                newim=tmp[0]            
                print ('{} objects detected in plane'.format(len(tmp[1])))
            else:
                for i in range(len(newim)):
                    tmp = find_cells.cell_detect(im[i].astype('uint16'), cell_area_thresh=cell_area_thresh, cell_circularity_thresh=cell_circularity_thresh, abs_intensity_percentile_thresh=abs_intensity_percentile_thresh, xyz_scale=xyz_scale, edge_finding_param=edge_finding_param, sigma=sigma, dilation_kernel_size=dilation_kernel_size)         
                    newim[i,...] = tmp[0]
                    print ('{} objects detected in plane {}'.format(len(tmp[1]), i))

                
    if show_final == True:
        sitk.Show(sitk.GetImageFromArray(newim))
    
    return newim

    
#%%
import os
from skimage.external import tifffile
import cv2
import time



def full_sizeddatafld_loader(cores = 10, resize_save = False, injdetect = False, conv_net_celldetect = False, full_sizedatafld=None, outdr = None, resizefactor = None, xyz_scale = None, injcoordinatesfld = None, **kwargs):
    '''Function to load files from full sized data folder, used when input data has already been deleted.
    '''
    start = time.time()    
    
    if full_sizedatafld == None:
        full_sizedatafld = kwargs['full_sizedatafld']
    
    if outdr == None:
        outdr = kwargs['outputdirectory']
    
    if resizefactor == None:
        resizefactor = kwargs['resizefactor']
    

    #files to load
    fl_to_process = [os.path.join(full_sizedatafld, subfld, xx) for subfld in [zz for zz in os.listdir(full_sizedatafld) if '.txt' not in zz] for xx in os.listdir(os.path.join(full_sizedatafld, subfld))]

    #make folders (done at this step to prevent parallelization issues)
    svloclst=[os.path.join(outdr, fl[fl.rfind('full_sizedatafld/') + 17: fl.rfind('/') - 5] +'_resized_ch' + fl[fl.rfind('C') + 1: fl.rfind('C') + 3]) for fl in fl_to_process]
    svlocs = []; [svlocs.append(xx) for xx in svloclst if xx not in svlocs]
    [makedir(svloc) for svloc in svlocs]
    
    #set up parellel pool    
    try:
        p
    except NameError:
        p=mp.Pool(cores)
        
    #perform action
    if resize_save != False:
        iterlst=[]; [iterlst.append((outdr, fl, int(resizefactor))) for fl in fl_to_process]  
        p.map(load_resize_save, iterlst);
        p.terminate()

    #detect injection site per plane        
    elif injdetect != False:
        print ('here we go with injdetect...')        

        if xyz_scale == None:
            try:            
                xyz_scale = kwargs['xyzscale']
            except NameError:
                print ('no xyz_scale, using (1.63, 1.63, 3)' )                
                xyz_scale = (1.63, 1.63, 3)
        if injcoordinatesfld == None:
            injcoordinatesfld = os.path.join(full_sizedatafld[:full_sizedatafld.rfind('/')], 'injection', 'injcoordinatesfld')
            
        #make folder if not there                
        makedir(injcoordinatesfld)        
        
        #run everything
        iterlst=[]; [iterlst.append((cores, injcoordinatesfld, xyz_scale, fl)) for fl in fl_to_process]  
        p.map(inject_detect, iterlst);
        p.terminate()
        return
        
    elif conv_net_celldetect != False:    
        print ('Waiting for Conv Net')
    
    else:
        iterlst=[]; [iterlst.append((fl, resizefactor)) for fl in fl_to_process]  
        p.map(load_resize_save, iterlst);
        p.terminate()
    
    print ('Time taken: {} minutes'.format((time.time() - start) / 60))
    return svlocs
    
def load_resize_save((outdr, fl, resizefactor)):

    #load    
    svloc=os.path.join(outdr, fl[fl.rfind('full_sizedatafld/') + 17: fl.rfind('/') - 5] +'_resized_ch' + fl[fl.rfind('C') + 1: fl.rfind('C') + 3])    
    im = tifffile.imread(fl) 
    #resize    
    if len(im.shape) == 2:  #grayscale images  
        y,x = im.shape
        xfct = int(x/resizefactor)
        yfct = int(y/resizefactor)
    elif len(im.shape) == 3: #color images from cell detect
        y,x,c = im.shape
        xfct = int(x/resizefactor)
        yfct = int(y/resizefactor)
    im1 = cv2.resize(im, (int(xfct), int(yfct)), interpolation=cv2.INTER_AREA)
    tifffile.imsave(os.path.join(svloc, fl[fl.rfind('/')+1:]), im1)        
    return
    
    
def tiffcombiner_new(vol_to_process, **kwargs):
    lst=os.listdir(vol_to_process); lst1=[os.path.join(vol_to_process, fl) for fl in lst]; lst1.sort()
    imstack=tifffile.imread(lst1)
    print ('imstack shape before squeeze: {}'.format(imstack.shape))
    if len(imstack.shape) >3:    
        imstack=np.squeeze(imstack)    
    
    print ('imstack shape after squeeze: {}'.format(imstack.shape))
    #tifffile.imsave(dct[chnlst[chindex]]+'_horizontal.tif',imstack.astype('uint16'))        
    
    try: ###check for orientation differences, i.e. from horiztonal scan to sagittal for atlas registration       
        imstack=np.swapaxes(imstack, *kwargs['swapaxes'])  
    except:
        pass
    
    print ('imstack shape after reslize: {}'.format(imstack.shape))
    tifffile.imsave(vol_to_process + '.tif', imstack.astype('uint16'))
    return
    
def inject_detect((cores, injcoordinatesfld, xyz_scale, fl)):
    '''wrapper function to allow for parellelization of inj detect on local machine
    '''
    #make inputs for find_inj_processor
    #stitchdct = {fl[fl.rfind('C')+1:fl.rfind('C')+3] : tifffile.imread(fl) } #dct of "ch":np.array(image)
    #zpln = fl[fl.rfind('Z')+1:fl.rfind('.')]

    #run
    #find_inj_processor(cores, stitchdct, zpln, injcoordinatesfld, xyz_scale)

    return
#%% wrappers    
def load_combine(outdr):
    '''wrapper function to load files and combine them
    '''    
    full_sizedatafld = os.path.join(outdr, 'full_sizedatafld')
    kwargs={}
    with open(os.path.join(outdr, 'param_dict.p'), 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    resizefactor = int(kwargs['resizefactor'])
    xyz_scale = kwargs['xyz_scale']
    
    start = time.time()  
    
    #save plns
    svlocs = full_sizeddatafld_loader(cores = 10, resize_save = True, injdetect = False, conv_net_celldetect = False, full_sizedatafld=full_sizedatafld, outdr = outdr, resizefactor = resizefactor, xyz_scale = xyz_scale)
    
    #combine tiffs
    [tiffcombiner_new(vol_to_process, **kwargs) for vol_to_process in svlocs]
    
    #output
    print ('FULL JOB: Time taken: {} minutes'.format((time.time() - start) / 60))        
    return
#%%
def inject_detect_wrapper(outdr):
    '''wrapper function to load files, inj detect, and save .np files
    '''    
    full_sizedatafld = os.path.join(outdr, 'full_sizedatafld')
    kwargs={}
    with open(os.path.join(outdr, 'param_dict.p'), 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    resizefactor = int(kwargs['resizefactor'])
    xyz_scale = kwargs['xyz_scale']
    
    start = time.time()  
    
    #save plns
    full_sizeddatafld_loader(cores = 10, resize_save = False, injdetect = True, conv_net_celldetect = False, full_sizedatafld=full_sizedatafld, outdr = outdr, resizefactor = resizefactor, xyz_scale = xyz_scale)
    
    
    #output
    print ('FULL JOB: Time taken: {} minutes'.format((time.time() - start) / 60))        
    return    
    
    


#%%    


        
#%%
        
def update_inj_sites(pths, cores=6, threshold = 0.035, num_labels_to_keep=1, resampledforelastix = True, masking = True):
    '''Function to update inj site quantification - NOTE THIS IS FOR POINT TRANSFORM (i.e. taking pixels)
    '''
    goodlst=[]; bdlst=[]    
    
    if type(pths) == dict or type(pths) == collections.OrderedDict: pths = pths.values()
       
    for outdr in pths:
        try:
            sys.stdout.write('\n\n\n***Updating injection site for {}....***\n\n'.format(outdr))        
            kwargs=load_kwargs(outdr)
            
            #elastix        
            transformfile = make_inverse_transform([xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0], cores = cores, **kwargs)
            
            #detect injection site
            array = inj_detect_using_labels(threshold = threshold, resampledforelastix = resampledforelastix, num_labels_to_keep=num_labels_to_keep, show = False, save = True, masking = masking, **kwargs)
            
            #apply resizing point transform
            txtflnm = point_transform_due_to_resizing(array, chtype = 'injch', **kwargs)
    
            #run transformix on points
            points_file = point_transfomix(txtflnm, transformfile, chtype = 'injch')
    
            #convert registered points into structure counts
            transformed_pnts_to_allen(points_file, ch_type = 'injch', colname = 'injection_site_pixels', point_or_index=None, **kwargs)
            
            goodlst.append(outdr)
        except:             
            bdlst.append(outdr)

        
    sys.stdout.write('\n\nGood paths were: {}'.format(goodlst))
    sys.stdout.write('\n\nBad paths were: {}'.format(bdlst))
            
    return


if __name__ == '__main__':

    #set pth
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    
    #just find all brains:
    allbrainpths = find_all_brains(tracing_output_fld)    
    pth = [xx for xx in allbrainpths if 'bl6_ts' not in xx and 'aav' not in xx]
    
    #crii = [xx for xx in allbrainpths if 'bl6_crII/' in xx]
    #crii.remove('/home/wanglab/wang/pisano/tracing_output/bl6_crII/20160628_bl6_crii_250r_03_almostnolabel') #analysis broke on this, wait for rerun
    #update injection sites:    
    update_inj_sites(pth, cores = 11, threshold = 0.15, num_labels_to_keep = 1, resampledforelastix = True, masking = True)




#%%
    

    
#%%
def cellcount_arrayjob(jobid, outdr = False, jobfactor = False, cell_area_thresh=[230, 1100], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.2, **kwargs):
    '''Used to rerun cell count jobs. Note this function assumes you have already deleted previous cell counts
    
    Inputs
    --------------------------------
    jobid: int, used for array jobs
    outdr: (optional) folder of lightsheet - if nonfalse this will override (reload) **kwargs
    jobfactor: (optional number of planes to process/per job)
    '''    
    #testing: outdr = '/home/wanglab/wang/pisano/tracing_output/bl6_crI/db_20160616_cri_53hr'
    ##handle inputs
    if outdr: 
        kwargs = load_kwargs(outdr)
    else:
        kwargs = load_kwargs(kwargs['outputdirectory'])
            
    ############################vars
    if jobfactor: 
        sf = jobfactor
    else:
        sf = int(kwargs['slurmjobfactor'])
    
    ###process planes
    for job in [(jobid*sf)+x for x in range(sf)]:
        zpln = str(job).zfill(4)
        ####### 
        vol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch'][0]
        if zpln == '0000': writer(vol.full_sizedatafld, 'Updating Cell detection using tools.utils.cellcount_arrayjob\n\nParameters used are: \n   xyz_scale: {}\n   cell_area_thresh: {}\n   cell_circularity_thresh: {}\n   abs_intensity_percentile_thresh: {}\n\n'.format(vol.xyz_scale, cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh), flnm='cell_count_rerun.txt'); sys.stdout.flush()
        try:            
            vol.zdct[zpln] #dictionary of files for single z plane
        except KeyError:
            return 'ArrayJobID/SF exceeds number of planes'
        
        #####################Cell detect ###
        try:     
            #find_cells_processor(cores, stitchdct, zpln, vol.cellcoordinatesfld, vol.xyz_scale) #find cells and return numpy arrays
            #load image        
            im = tifffile.imread([xx for xx in listdirfull(vol.full_sizedatafld_vol) if 'Z{}.tif'.format(zpln) in xx][0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    
                disp, centers, contours=find_cells.cell_detect(im.astype('uint16'), xyz_scale=vol.xyz_scale, cell_area_thresh=cell_area_thresh, cell_circularity_thresh=cell_circularity_thresh, abs_intensity_percentile_thresh=abs_intensity_percentile_thresh)
                zyx_array=np.asarray([(int(zpln), centers[i][0], centers[i][1])for i in range(len(centers))])        
                zyxcenters_contours=np.asanyarray(zip(zyx_array, contours)) ###2d array where 0d=zyx of centers, and 1d=contours
                np.save(os.path.join(vol.cellcoordinatesfld, 'ch{}_cells_zyxcenters_contours_Z{}.npy'.format(vol.channel, zpln)), zyxcenters_contours)
                writer(vol.full_sizedatafld, 'Processed zpln {}\n'.format(zpln), flnm='cell_count_rerun.txt'); sys.stdout.flush()
        except:
            print ('cell detection failed...')
        ######################################################################  
    return
def cellcount_arrayjob_par(cores, outdr, jobfactor = 100, cell_area_thresh=[230, 1100], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.2):
    
    kwargs = load_kwargs(outdr)

    
    p = mp.Pool(cores)
    iterlst = []; [iterlst.append((jobid, outdr, jobfactor, cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh, kwargs)) for jobid in range(23)]
    p.map(cellcount_arrayjob_helper, iterlst)
    
    sys.stdout.write('Completed cell detection for {}'.format(outdr)); sys.stdout.flush()
    
    return
    
def cellcount_arrayjob_helper((jobid, outdr, jobfactor, cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh, kwargs)):
    '''Used to rerun cell count jobs. Note this function assumes you have already deleted previous cell counts
    
    Parellized version
    
    Inputs
    --------------------------------
    jobid: int, used for array jobs
    outdr: (optional) folder of lightsheet - if nonfalse this will override (reload) **kwargs
    jobfactor: (optional number of planes to process/per job)
    '''    
    #testing: outdr = '/home/wanglab/wang/pisano/tracing_output/bl6_crI/db_20160616_cri_53hr'
    ##handle inputs
         
    ############################vars
    if jobfactor: 
        sf = jobfactor
    else:
        sf = int(kwargs['slurmjobfactor'])
    
    ###process planes
    for job in [(jobid*sf)+x for x in range(sf)]:
        zpln = str(job).zfill(4)
        ####### 
        vol = [xx for xx in kwargs['volumes'] if xx.ch_type == 'cellch'][0]
        if zpln == '0000': writer(vol.full_sizedatafld, 'Updating Cell detection using tools.utils.cellcount_arrayjob_par\n\nDirectory is {}\nParameters used are: \n   xyz_scale: {}\n   cell_area_thresh: {}\n   cell_circularity_thresh: {}\n   abs_intensity_percentile_thresh: {}\n\n'.format(outdr, vol.xyz_scale, cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh), flnm='cell_count_rerun.txt'); sys.stdout.flush()
        try:            
            vol.zdct[zpln] #dictionary of files for single z plane
        except KeyError:
            return 'ArrayJobID/SF exceeds number of planes'
        
        #####################Cell detect ###
        try:     
            #find_cells_processor(cores, stitchdct, zpln, vol.cellcoordinatesfld, vol.xyz_scale) #find cells and return numpy arrays
            #load image        
            im = tifffile.imread([xx for xx in listdirfull(vol.full_sizedatafld_vol) if 'Z{}.tif'.format(zpln) in xx][0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    
                disp, centers, contours=find_cells.cell_detect(im.astype('uint16'), xyz_scale=vol.xyz_scale, cell_area_thresh=cell_area_thresh, cell_circularity_thresh=cell_circularity_thresh, abs_intensity_percentile_thresh=abs_intensity_percentile_thresh)
                zyx_array=np.asarray([(int(zpln), centers[i][0], centers[i][1])for i in range(len(centers))])        
                zyxcenters_contours=np.asanyarray(zip(zyx_array, contours)) ###2d array where 0d=zyx of centers, and 1d=contours
                np.save(os.path.join(vol.cellcoordinatesfld, 'ch{}_cells_zyxcenters_contours_Z{}.npy'.format(vol.channel, zpln)), zyxcenters_contours)
                writer(vol.full_sizedatafld, 'Processed zpln {}\n'.format(zpln), flnm='cell_count_rerun.txt'); sys.stdout.flush()
        except:
            print ('cell detection failed...')
        ######################################################################  
    return

    
def rerun_cellcount_registration_transformation(cores, outdr, cleanfolder = False):
    '''Function to rerun cell detection, registration and transformation
    
    Inputs:
        cores = # for parellization
        outdr = location of folder from lightsheet
        cleanfolder = 
                    False = doesn't clean folder will overwrite info, this can get messy
                    'deletefiles' = removes old folders, data before new ones
                    'movefiles' =  moves files into an new directory called 'old'
    '''
    #inputs
    kwargs = load_kwargs(outdr)
    
    #rerun cell counts, ONLY EDGE DETECTION IS FUNCTIONAL NOW:
    if cleanfolder: clean_folder(outdr, deletefiles=cleanfolder)
    cellfld = os.path.join(outdr, 'cells'); makedir(cellfld); makedir(cellfld+'/cellcoordinatesfld'); makedir(cellfld+'celldetect3d')
    cellcount_arrayjob_par(cores, outdr, jobfactor = 100, cell_area_thresh=[230, 1100], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.2)
    
    #detect in 3d:
    for jobid in range(75):
        detect_cells_in_3d(jobid, cores=cores, mxdst=30, pln_chnk=30, ovlp_plns=25, **kwargs)
    
    #register and transform:
    try:
        for jobid in range(3):
            identify_structures_w_contours(jobid, cores=cores, make_color_images=False, consider_only_multipln_contours=True, overlay_on_original_data=False, **kwargs)
    except IndexError:
        #stitch images if necessary
        for jobid in range(3):
            try:
                tiffcombiner_new(jobid, **kwargs)
            except OSError:
                pass
        #resample
        vols = kwargs['volumes']
        for vol in vols:
            vol.add_resampled_for_elastix_vol(vol.downsized_vol+'_resampledforelastix.tif')
            resample_par(cores, vol.downsized_vol+'.tif', kwargs['AtlasFile'], svlocname=vol.resampled_for_elastix_vol, singletifffile=True, resamplefactor=1.3)
        for jobid in range(3):
            identify_structures_w_contours(jobid, cores=cores, make_color_images=False, consider_only_multipln_contours=True, overlay_on_original_data=False, **kwargs)    
    return
    
    

