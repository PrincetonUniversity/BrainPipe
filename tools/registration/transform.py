# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:26:52 2016

@author: wanglab
"""

from tools.utils.io import makedir, removedir, chunkit, writer, load_kwargs
import multiprocessing as mp
import pandas as pd
import numpy as np
from math import ceil
import re, sys, os, cv2, gc, shutil, random
import pickle
import tifffile
from tools.imageprocessing.preprocessing import resample_par
from tools.registration.register import make_inverse_transform, point_transformix
import subprocess as sp
import SimpleITK as sitk
from collections import Counter

##FIXME: need to fix orientation!!
###for a contour, apply a color variable, ungroup contours, sort by z. Array job to mark contours and then save
def ovly_3d(ch, svlc, tplst, valid_plns, outdr, vol_to_process, resizefactor, contour_class_lst, multiplane, ovly, core, cores):
    '''function to apply independent colors to contours that have been detected on multiple planes, and save images. 
    This has been parallelized and calculates a range based on the number of cores.
    '''
    ####assign random color to each group of contours, the break up groups, and sort by plane    
    #use this for looking at multi contours only
    if multiplane != False:       
        multilist=[x for x in tplst if len(x.plns) >1]
        tplst=multilist
    contour_color_lst=[]        
    for contours in tplst:            
        color=(random.randint(50,255), random.randint(50,255), random.randint(50,255))
        for plns in contours.plns.itervalues():
            contour_color_lst.append([plns, color]) ###break apart list        
    #sort by z
    try: #sort by z
        contour_color_lst.sort()
    except ValueError: #done to suppress error
        pass
    ############################################load find full size files############################################
    makedir(svlc)
    dr=vol_to_process.full_sizedatafld_vol
    #dr='/home/wanglab/wang/pisano/tracing_output/H129_EGFP_NLS_vc22_01_lob5_intensityrescale/full_sizedatafld/ch01'
    fl=[f for f in os.listdir(dr) if '.tif' in f]
    reg=re.compile(r'(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.tif)') ###regex used since code changes file format
    matches=map(reg.match, fl)
    ##find index of z,y,x,ch in matches
    z_indx=matches[0].span('z')
    ####make dictionary where each pln is a key, values are all point that should be drawn
    zdct={}
    for i in range(len(contour_color_lst)):
        pln=contour_color_lst[i] ##check
        try:        
            zdct[str(pln[0][0]).zfill(4)].append(pln)
        except KeyError:
            zdct[str(pln[0][0]).zfill(4)]=[]
            zdct[str(pln[0][0]).zfill(4)].append(pln)
    ############parse jobs:
    chnkrng=chunkit(core, cores, zdct)
    ########### apply points to each pln
    if ovly == True:    ###ovly 3d contours onto data
        flpth=''.join(matches[0].groups())
        y,x=tifffile.imread(os.path.join(dr, flpth)).shape
        dsf=resizefactor
        for i in range(chnkrng[0], chnkrng[1]):
            try:        
                pln=zdct.keys()[i]; cntrs=zdct[zdct.keys()[i]]
                ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs)
                del cntrs, pln; gc.collect()
            except:
                pass
        ###make zpln range and make empty planes on the zplns where no contours are found    
        try:
            plnlst=[int(plns) for plns in zdct.iterkeys()]; plnrng=range(min(plnlst), max(plnlst)+1)
            nocontour_plns=set(plnrng).difference(set(plnlst))
                ############parse jobs:  
            chnkrng=chunkit(core, cores, nocontour_plns)
            print ('\n\ncontours not found on planes: {}\nmaking empty images for those planes...'.format(nocontour_plns))
            for i in range(chnkrng[0], chnkrng[1]):
                pln=list(nocontour_plns)[i]
                blnk_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln)
                del pln; gc.collect()
            return 
        except ValueError: #no contours found
            print ('No contour found in ch{}, not making color stack...'.format(ch))
            return 
    elif ovly == False:  ###make downsized 3d contours
        flpth=''.join(matches[0].groups())
        y,x=tifffile.imread(os.path.join(dr, flpth)).shape
        dsf=resizefactor
        for i in range(chnkrng[0], chnkrng[1]):
            try:        
                pln=zdct.keys()[i]; cntrs=zdct[zdct.keys()[i]]
                no_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs)
                del cntrs, pln; gc.collect()
            except:
                pass
        ###make zpln range and make empty planes on the zplns where no contours are found    
        try:
            plnlst=[int(plns) for plns in zdct.iterkeys()]; plnrng=range(min(plnlst), max(plnlst)+1)
            nocontour_plns=set(plnrng).difference(set(plnlst))
            if len(nocontour_plns) != 0:
                print ('\n\ncontours not found on planes: {}\nmaking empty images for those planes...'.format(nocontour_plns))
                ############parse jobs:  
            chnkrng=chunkit(core, cores, nocontour_plns)
            for i in range(chnkrng[0], chnkrng[1]):
                try:                
                    pln=list(nocontour_plns)[i]
                    blnk_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln)
                    del pln; gc.collect()
                except IndexError:
                    pass
            return 
        except ValueError: #no contours found
            print ('No contour found in ch{}, not making color stack...'.format(ch))
            return 

def ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=cv2.imread(os.path.join(dr, flpth1), 1)
    for cntr in cntrs:            
        cv2.fillConvexPoly(im, cntr[0][3], color=cntr[1])
    tifffile.imsave(os.path.join(svlc, '3DcontourS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('processed pln {}, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs, im
    gc.collect()
    return

def blnk_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln):
    flpth1=flpth[:z_indx[0]]+str(pln).zfill(4)+flpth[z_indx[1]:]
    im=cv2.imread(os.path.join(dr, flpth1), 1)
    tifffile.imsave(os.path.join(svlc, '3DcontourS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('No contours found on pln {}, making empty file, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, im
    gc.collect()
    return

def no_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=np.zeros((y,x))
    for cntr in cntrs:            
        cv2.fillConvexPoly(im, cntr[0][3], color=cntr[1])
    tifffile.imsave(os.path.join(svlc, '3DcontourS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('processed pln {}, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs, im
    gc.collect()
    return

def blnk_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=np.zeros((y,x))
    tifffile.imsave(os.path.join(svlc, '3DcontourS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('No contours found on pln {}, making empty file, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, im
    gc.collect()
    return


def identify_structures_w_contours(jobid, cores=5, make_color_images=False, overlay_on_original_data=False, consider_only_multipln_contours=False, **kwargs):
    '''function to take 3d detected contours and apply elastix transform
    '''    
    #######################inputs and setup#################################################
    ###inputs    
    kwargs = load_kwargs(**kwargs)
    outdr=kwargs['outputdirectory']
    vols=kwargs['volumes']
    reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]

    ###get rid of extra jobs
    if jobid >= len([xx for xx in vols if xx.ch_type != 'regch']): ###used to end jobs if too many are called
        print ('jobid({}) >= volumes {}'.format(jobid, len([xx for xx in vols if xx.ch_type != 'regch'])))         
        return    
    
    ###volumes to process: each job represents a different contour volume
    vol_to_process=[xx for xx in vols if xx.ch_type != 'regch'][jobid]
    ch=vol_to_process.channel
    print(vol_to_process.ch_type)

    #find appropriate folders for contours
    if vol_to_process.ch_type == 'cellch':
        detect3dfld = reg_vol.celldetect3dfld
        coordinatesfld = reg_vol.cellcoordinatesfld
    elif vol_to_process.ch_type == 'injch':
        detect3dfld = reg_vol.injdetect3dfld
        coordinatesfld = reg_vol.injcoordinatesfld
    
    #set scale and atlas
    xscl, yscl, zscl = reg_vol.xyz_scale ###micron/pixel  
    zmx, ymx, xmx = reg_vol.fullsizedimensions
    AtlasFile=reg_vol.atlasfile
    print ('Using {} CORES'.format(cores))
    try:
        p
    except NameError:
        p=mp.Pool(cores)
    resizefactor=kwargs['resizefactor']
    brainname=reg_vol.brainname

    ############################################################################################################
    #######################use regex to sort np files by ch and then by zpln####################################
    ############################################################################################################    
    fl=[f for f in os.listdir(detect3dfld) if '.p' in f and 'ch' in f] #sorted for raw files
    reg=re.compile(r'(.*h+)(?P<ch>\d{2})(.*)(.p)')
    matches=map(reg.match, fl)      

    ##load .np files
    sys.stdout.write('\njobid({}), loading ch{} .p files to extract contour_class objects....'.format(jobid, ch))
    contour_class_lst=[]
    for fl in [os.path.join(detect3dfld, ''.join(xx.groups())) for xx in matches if xx.group('ch')[-2:] in ch]:
        tmpkwargs={}
        pckl=open(fl, 'rb'); tmpkwargs.update(pickle.load(pckl)); pckl.close()
        if consider_only_multipln_contours == False:
            tmplst=tmpkwargs['single']
            [tmplst.append(xx) for xx in tmpkwargs['multi']]
        elif consider_only_multipln_contours == True:    
            tmplst=tmpkwargs['multi']
        [contour_class_lst.append(xx) for xx in tmplst]
    sys.stdout.write('\ndone loading contour_class objects.\n')  
    
    #check for successful loading
    if len(contour_class_lst) == 0:
        print ('Length of contours in ch{} was {}, ending process...'.format(jobid, len(contour_class_lst)))        
        try:
            p.terminate()
        except:
            1
        return

    ############################################################################################################        
    ##############################make color files##############################################################
    ############################################################################################################    
    if make_color_images == True:
        sys.stdout.write('\nmaking 3d planes...')
        sys.stdout.flush()
        valid_plns=range(0, zmx+1)        
        svlc=os.path.join(outdr, 'ch{}_3dcontours'.format(ch)); removedir(svlc)       
        if overlay_on_original_data == False:
            ovly = False
        elif overlay_on_original_data == True:
            ovly = True
        iterlst=[]; [iterlst.append((ch, svlc, contour_class_lst, valid_plns, outdr, vol_to_process, resizefactor, contour_class_lst, consider_only_multipln_contours, ovly, core, cores)) for core in range(cores)]
        p.starmap(ovly_3d, iterlst);
        lst=os.listdir(svlc); lst1=[os.path.join(svlc, xx) for xx in lst]; lst1.sort(); del lst; del iterlst
        ###load ims and return dct of keys=str(zpln), values=np.array          
        sys.stdout.write('\n3d planes made, saved in {},\nnow compressing into single tifffile'.format(svlc))        
        imstack=tifffile.imread(lst1); del lst1
        if len(imstack.shape) > 3:    
            imstack=np.squeeze(imstack)    
        try: ###check for orientation differences, i.e. from horiztonal scan to sagittal for atlas registration       
            imstack=np.swapaxes(imstack, *kwargs['swapaxes'])  
        except:
            pass
        tiffstackpth=os.path.join(outdr, '3D_contours_ch{}_{}'.format(ch, brainname))
        tifffile.imsave(tiffstackpth,imstack.astype('uint16')); del imstack; gc.collect()
        shutil.rmtree(svlc)
        sys.stdout.write('\ncolor image stack made for ch{}'.format(ch))
    else:
        sys.stdout.write('\nmake_color_images=False, not creating images')        
    ############################################################################################################        
    ######################apply point transform and make transformix input file#################################
    ############################################################################################################   
    ###find centers and add 1's to make nx4 array for affine matrix multiplication to account for downsizing
    ###everything is in PIXELS
    contourarr=np.empty((len(contour_class_lst),3));
    for i in range(len(contour_class_lst)):
        contourarr[i,...]=contour_class_lst[i].center ###full sized dimensions: if 3x3 tiles z(~2000),y(7680),x(6480) before any rotation 
    try:        
        contourarr=swap_cols(contourarr, *kwargs['swapaxes']) ###change columns to account for orientation changes between brain and atlas: if horizontal to sagittal==>x,y,z relative to horizontal; zyx relative to sagittal
        z,y,x=swap_cols(np.array([vol_to_process.fullsizedimensions]), *kwargs['swapaxes'])[0]##convert full size cooridnates into sagittal atlas coordinates
        sys.stdout.write('\nSwapping Axes')
    except: ###if no swapaxes then just take normal z,y,x dimensions in original scan orientation
        z,y,x=vol_to_process.fullsizedimensions
        sys.stdout.write('\nNo Swapping of Axes')
    d1,d2=contourarr.shape
    nx4centers=np.ones((d1,d2+1))
    nx4centers[:,:-1]=contourarr
    ###find resampled elastix file dim
    
    print(os.listdir(outdr))
    print([xx.channel for xx in vols if xx.ch_type == 'regch'])
    with tifffile.TiffFile([os.path.join(outdr, f) for f in os.listdir(outdr) if 'resampledforelastix' in f and 'ch{}'.format([xx.channel for xx in vols if xx.ch_type == 'regch'][0]) in f][0]) as tif:  
        zr=len(tif.pages)
        yr,xr=tif.pages[0].shape
        tif.close()
    ####create transformmatrix
    trnsfrmmatrix=np.identity(4)*(zr/z, yr/y, xr/x, 1) ###downscale to "resampledforelastix size"
    sys.stdout.write('trnsfrmmatrix:\n{}\n'.format(trnsfrmmatrix))
    #nx4 * 4x4 to give transform
    trnsfmdpnts=nx4centers.dot(trnsfrmmatrix) ##z,y,x
    sys.stdout.write('first three transformed pnts:\n{}\n'.format(trnsfmdpnts[0:3]))
    #create txt file, with elastix header, then populate points
    txtflnm='{}_zyx_transformedpnts_ch{}.txt'.format(brainname, ch)
    pnts_fld=os.path.join(outdr, 'transformedpoints_pretransformix'); makedir(pnts_fld)
    transforminput=os.path.join(pnts_fld, txtflnm)
    removedir(transforminput)###prevent adding to an already made file
    writer(pnts_fld, 'index\n{}\n'.format(len(trnsfmdpnts)), flnm=txtflnm)    
    sys.stdout.write('\nwriting centers to transfomix input points text file: {}....'.format(transforminput))
    stringtowrite = '\n'.join(['\n'.join(['{} {} {}'.format(i[2], i[1], i[0])]) for i in trnsfmdpnts]) ####this step converts from zyx to xyz*****
    writer(pnts_fld, stringtowrite, flnm=txtflnm)    
    #[writer(pnts_fld, '{} {} {}\n'.format(i[2],i[1],i[0]), flnm=txtflnm, verbose=False) for i in trnsfmdpnts] ####this step converts from zyx to xyz*****
    sys.stdout.write('...done writing centers.'); sys.stdout.flush()
    del trnsfmdpnts, trnsfrmmatrix, nx4centers, contourarr; gc.collect()
    ############################################################################################################        
    ####################################elastix for inverse transform###########################################
    ############################################################################################################     
    transformfile=make_inverse_transform(vol_to_process, cores, **kwargs)
    assert os.path.exists(transformfile)
    sys.stdout.write('\n***Elastix Inverse Transform File: {}***'.format(transformfile))
    ############################################################################################################        
    ####################################transformix#############################################################
    ############################################################################################################        
    if make_color_images != False:
        #apply transform to 3d_tiffstack:
        transformimageinput=tiffstackpth; elastixpth=os.path.join(outdr, 'elastix')
        trnsfrm_outpath=os.path.join(elastixpth, '3D_contours_ch{}_{}'.format(ch, brainname)); makedir(trnsfrm_outpath)
        writer(trnsfrm_outpath,'\nProcessing ch{} 3D...'.format(ch))
        #transformfiles=[os.path.join(elastixpth, xx) for xx in os.listdir(os.path.join(outdr, 'elastix')) if "TransformParameters" in xx]; mxx=max([xx[-5] for xx in transformfiles])
        #transformfile=os.path.join(elastixpth, 'TransformParameters.{}.txt'.format(mxx))
        trnsfrm_out_file = os.path.join(trnsfrm_outpath, 'result.tif') #output of transformix
        transformimageinput_resized=transformimageinput[:-4]+'_resampledforelastix.tif'
        print ('Resizing {}'.format(transformimageinput_resized))        
        resample_par(cores, transformimageinput, AtlasFile, svlocname=transformimageinput_resized, singletifffile=True, resamplefactor=1.7)                
        sp.call(['transformix', '-in', transformimageinput_resized, '-out', trnsfrm_outpath, '-tp', transformfile])
        writer(trnsfrm_outpath,'\n   Transformix File Generated: {}'.format(trnsfrm_out_file))
        writer(trnsfrm_outpath,'\n   Passing colorcode: {} file as {}'.format(ch, os.path.join(trnsfrm_outpath, 'depthcoded.png')))
        ###depth coded image of transformix result; not functional yet
        #depth.colorcode(trnsfrm_out_file, trnsfrm_outpath)
        #getvoxels(trnsfrm_out_file, os.path.join(trnsfrm_outpath, 'zyx_voxels_{}.npy'.format(ch)))
        #allen_compare(AtlasFile, svlc, trnsfrm_outpath)
        ##if successful delete contour cooridnates and maybe contourdetect3d flds
    ############################################################    
    
    ##############apply transform to points#####################
    elastixpth=os.path.join(outdr, 'elastix_inverse_transform')
    trnsfrm_outpath=os.path.join(elastixpth, 'ch{}_3dpoints'.format(ch)); makedir(trnsfrm_outpath)
    writer(trnsfrm_outpath,'\n***********Starting Transformix for: {}***********'.format(ch)); sys.stdout.flush()
    #transformfiles=[os.path.join(elastixpth, xx) for xx in os.listdir(os.path.join(outdr, 'elastix')) if "TransformParameters" in xx]; mxx=max([xx[-5] for xx in transformfiles])
    #transformfile=os.path.join(elastixpth, 'TransformParameters.{}.txt'.format(mxx))
    trnsfrm_out_file = os.path.join(trnsfrm_outpath, 'outputpoints.txt') #MIGHT NEED TO CHANGE THIS
    sp.call(['transformix', '-def', transforminput, '-out', trnsfrm_outpath, '-tp', transformfile])
    #sp.call(['transformix', '-def', 'all', '-out', trnsfrm_outpath, '-tp', transformfile]) ##displacement field
    writer(trnsfrm_outpath,'\n   Transformix File Generated: {}'.format(trnsfrm_out_file))
    ####################################################################################    
    ##############generate list and image overlaid onto allen atlas#####################
    ####################################################################################    
    name = 'job{}_{}'.format(jobid, vol_to_process.ch_type)    
    transformed_pnts_to_allen(trnsfrm_out_file, ch, cores, name=name, **kwargs)
    writer(outdr, '*************STEP 5*************\n Finished')    
    print ('end of script')
    try:
        p.terminate()
    except:
        1
    return

def points_resample(src, original_dims, resample_dims, verbose = False):
    '''Function to adjust points given resizing by generating a transform matrix
    
    ***Assumes ZYX and that any orientation changes have already been done.***
    
    src: numpy array or list of np arrays of dims nx3
    original_dims (tuple)
    resample_dims (tuple)
    '''
    src = np.asarray(src)
    assert src.shape[-1] == 3, 'src must be a nx3 array'
    
    #initialize
    d1,d2=src.shape
    nx4centers=np.ones((d1,d2+1))
    nx4centers[:,:-1]=src
    
    #acount for resampling by creating transformmatrix
    zr, yr, xr = resample_dims
    z, y, x = original_dims
    
    #apply scale diff
    trnsfrmmatrix=np.identity(4)*(zr/float(z), yr/float(y), xr/float(x), 1)
    if verbose: sys.stdout.write('trnsfrmmatrix:\n{}\n'.format(trnsfrmmatrix))
    
    #nx4 * 4x4 to give transform
    trnsfmdpnts=nx4centers.dot(trnsfrmmatrix) ##z,y,x
    if verbose: sys.stdout.write('first three transformed pnts:\n{}\n'.format(trnsfmdpnts[0:3]))

    return trnsfmdpnts



def points_transform(src, dst, transformfile, verbose=False):
    '''Function to apply a tranform given a numpy.
    
    ***Assumes ZYX and that any orientation changes have already been done.***
    
    src: numpy array or list of np arrays of dims nx3
    dst: folder to save
    '''
    src = np.asarray(src)
    assert src.shape[-1] == 3, 'src must be a nx3 array'
    
    #create txt file, with elastix header, then populate points
    makedir(dst)
    pnts_fld=os.path.join(dst, 'transformedpoints_pretransformix'); makedir(pnts_fld)
    transforminput=os.path.join(pnts_fld, 'zyx_transformedpnts.txt')
    
    #prevent adding to an already made file
    removedir(transforminput)
    writer(pnts_fld, 'index\n{}\n'.format(len(src)), flnm=transforminput)
    if verbose: sys.stdout.write('\nwriting centers to transfomix input points text file: {}....'.format(transforminput))
    
    #this step converts from zyx to xyz*****
    stringtowrite = '\n'.join(['\n'.join(['{} {} {}'.format(i[2], i[1], i[0])]) for i in src])
    writer(pnts_fld, stringtowrite, flnm=transforminput)
    if verbose: sys.stdout.write('...done writing centers.'); sys.stdout.flush()
    
    #elastix for inverse transform
    trnsfrm_out_file = point_transformix(txtflnm = transforminput, dst = dst, transformfile = transformfile)
    assert os.path.exists(trnsfrm_out_file), 'Error Transform file does not exist: {}'.format(trnsfrm_out_file)
    #if verbose: sys.stdout.write('\n***Elastix Inverse Transform File: {}***'.format(transformfile))
    
    return trnsfrm_out_file
    
def transformed_pnts_to_allen(trnsfrm_out_file, ch, cores, point_or_index=None, name = False, **kwargs):
    '''function to take elastix point transform file and return anatomical locations of those points
    point_or_index=None/point/index: determines which transformix output to use: point is more accurate, index is pixel value(?)
    Elastix uses the xyz convention rather than the zyx numpy convention
    
    ###ASSUMES INPUT OF XYZ
    
    '''    
    #####inputs 
    assert type(trnsfrm_out_file)==str
    if point_or_index==None:
        point_or_index = 'OutputPoint'
    elif point_or_index == 'point':
        point_or_index = 'OutputPoint'
    elif point_or_index == 'index':
        point_or_index = 'OutputIndexFixed'
    try: #check to see if pool processes have already been spawned
        p
    except NameError:
        p=mp.Pool(cores)

    kwargs = load_kwargs(**kwargs)
    vols=kwargs['volumes']
    reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]

    ####load files
    id_table=pd.read_excel(os.path.join(kwargs['packagedirectory'], 'supp_files/id_table.xlsx')) ##use for determining neuroanatomical locations according to allen
    ann=sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile'])) ###zyx
    with open(trnsfrm_out_file, "rb") as f:                
        lines=f.readlines()
        f.close()
        
    #####populate post-transformed array of contour centers
    sys.stdout.write('\n{} points detected'.format(len(lines)))
    arr=np.empty((len(lines), 3))    
    for i in range(len(lines)):        
        arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z

    #optional save out of points
    np.save(kwargs['outputdirectory']+'/injection/zyx_voxels.npy', np.asarray([(z,y,x) for x,y,z in arr]))
        
    pnts=transformed_pnts_to_allen_helper_func(arr, ann); pnt_lst=[xx for xx in pnts if xx != 0]
    if len(pnt_lst)==0:
        raise ValueError('pnt_lst is empty')
    else:
        sys.stdout.write('\nlen of pnt_lst({})'.format(len(pnt_lst)))
    imstack = brain_structure_keeper(ann, True, *pnt_lst) ###annotation file, true=to depict density, list of pnts
    df=count_structure_lister(id_table, *pnt_lst)
    #########save out imstack and df
    nametosave = '{}{}_{}'.format(name, ch, reg_vol.brainname)
    tifffile.imsave(os.path.join(kwargs['outputdirectory'], nametosave + '_structure_density_map.tif'), imstack)
    excelfl=os.path.join(kwargs['outputdirectory'], nametosave + '_stuctures_table.xlsx')    
    df.to_excel(excelfl)
    print ('file saved as: {}'.format(excelfl))
    try:
        p.terminate()
    except:
        1
    return

       
def transformed_pnts_to_allen_helper_func(arr, ann, order = 'XYZ'):
    '''Function to transform given array of indices and return the atlas pixel ID from the annotation file
    
    Input
    --------------
    numpy array of Nx3 dimensions corresponding to ***XYZ*** coordinates
    ann = numpy array of annotation file
    order = 'XYZ' or 'ZYX' representing the dimension order of arr's input
    
    Returns
    -------------
    Pixel value at those indices of the annotation file, maintains order if NO BAD MAPPING
    '''        
    ########procecss
    pnt_lst=[]; badpntlst = []
    for i in range(len(arr)):
        try:        
            pnt=[int(x) for x in arr[i]]
            if order == 'XYZ': pnt_lst.append(ann[pnt[2], pnt[1], pnt[0]]) ###find pixel id; arr=XYZ; ann=ZYX
            elif order == 'ZYX': pnt_lst.append(ann[pnt[0], pnt[1], pnt[2]]) ###find pixel id; arr=ZYX; ann=ZYX
        except IndexError:
            badpntlst.append([pnt[2], pnt[1], pnt[0]]) #ZYX
            pass ######THIS NEEDS TO BE CHECKED BUT I BELIEVE INDEXES WILL BE OUT OF 
    sys.stdout.write('\n*************{} points do not map to atlas*********\n'.format(len(badpntlst))); sys.stdout.flush()
    return pnt_lst
        
def count_structure_lister(id_table, *args):
    '''Function that generates a pd table of structures where contour detection has been observed
    Inputs:
        id_table=annotation file
        *args=list of allen ID pixel values ZYX
    '''
    #make dictionary of pixel id:#num of the id
    cnt = Counter()
    for i in args:
        cnt[i]+=1
    
    #generate df + empty column
    if type(id_table) == str: id_table = pd.read_excel(id_table) #df=id_table.assign(count= [0]*len(id_table)) #add count columns to df
    df=id_table
    df['cell_count']=0
    
    #populate cell count in dataframe
    for pix_id, count in cnt.items():
        df.loc[df.id==pix_id, 'cell_count']=count

    return df
    
def structure_lister(id_table, *args):
    '''Function that returns a list of structures where contour detection has been observed
    Inputs:
        id_table=annotation file as np.array
        *args=list of allen ID pixel values
    '''
    df = id_table
    nmlst = []
    for i in args:
        nmlst.append(df.name[df.id==i])
    #regions_w_cells.append(df[df.cell_count>0])
    #return regions_w_cells
    return nmlst
         
    
def brain_structure_keeper(ann, depictdensity, *args):
    '''Function that generates an image of structures where contour detection has been observed
    Inputs:
        ann=annotation file as np.array
        depictdensity=True/False, if true normalizes contour counts per/region and normalizes them to upper half of 8bit pixel values
        *args=list of allen ID pixel values
    ''' 
    ###############################UNFINISHED    
    ##find zyx coordinates of args
    if depictdensity==True:        
        dct=Counter(args) #keys=number, #values=count
        mostcontoursinloc=max(dct.values())
        leastcontoursinloc=min(dct.values())
        #zip(np.linspace(leastcontoursinloc, mostcontoursinloc), np.logspace(127, 255))
        tick=0
        #pxrng=np.linspace(127, 255, int(len(dct)/2))       
        stack=np.zeros(ann.shape).astype('uint8') ##65000
        #b=[((count - np.mean(dct.values())) / np.std(dct.values())) for count in dct.itervalues()]
        for pixid, count in dct.iteritems():
            pixintensity= ((count - leastcontoursinloc ) / (mostcontoursinloc - leastcontoursinloc) * 255) #+ 127.5 ###scaling to upper half of pixel values; done to prevent zeroing out of minimum
            #pixintensity= (count - np.mean(dct.values())) / np.std(dct.values()); print pixintensity
            stack[ann==pixid] = pixintensity
            tick+=1
            print('Brain Strucutre keeper: done {} in {}'.format(tick, len(dct)))
        return stack
        ##################WORKING
    elif depictdensity == False:
        argss=list(set(args))
        stack=np.zeros(ann.shape).astype('uint8')
        for i in argss:
            stack[ann==i] = 255
        return stack

def swap_cols(arr, frm, to):
    '''helper function used to swap np array columns if orientation changes have been made pre-registration
    '''
    try:
        arr[:, [frm, to]]=arr[:, [to, frm]]
    except:
        print ('Array is likely empty - and so need to adjust thresholding')
    return arr


def detect_contours_in_3d_checker(jobid, pln_chnk=50, **kwargs):
    '''Not utilized yet
    '''
    kwargs = load_kwargs(**kwargs)
    outdr = kwargs['outputdirectory']
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]
    ###set volume to use
    vol=[xx for xx in vols if xx.ch_type != 'regch'][jobid]
    if vol.ch_type == 'cellch':
        detect3dfld = reg_vol.celldetect3dfld
        coordinatesfld = reg_vol.cellcoordinatesfld
    elif vol.ch_type == 'injch':
        detect3dfld = reg_vol.injdetect3dfld
        coordinatesfld = reg_vol.injcoordinatesfld    
    
      
    zmax = vols[0].fullsizedimensions[0]
    if len(os.listdir(detect3dfld)) != int(ceil(zmax / pln_chnk)):
        writer(outdr, '\n\n***************************STEP 4 FAILED*********************\n{} files found in {}. Should have {}.'.format(len(os.listdir(contourdetect3dfld)), contourdetect3dfld[contourdetect3dfld.rfind('/')+1:], int(ceil(zmax / pln_chnk))))
    else:
        writer(outdr, '\n\n***************************STEP 4 SUCCESS*********************\n{} files found in {}. Should have {}.'.format(len(os.listdir(contourdetect3dfld)), contourdetect3dfld[contourdetect3dfld.rfind('/')+1:], int(ceil(zmax / pln_chnk))))
    return

def filter_overinterpolated_pixels(arr, atlas=False, **kwargs):
    '''Remove pixels not matching the atlas
    '''
    if not atlas: atlas = tifffile.imread(kwargs['AtlasFile'])
    
    is_bad = (arr[:,0]>=atlas.shape[0]) | (arr[:,1]>=atlas.shape[1]) | (arr[:,2]>=atlas.shape[2])
    arr = arr[~is_bad]
    return arr
