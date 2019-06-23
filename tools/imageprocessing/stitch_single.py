#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:13:33 2017

FUNCTION TO STITCH FROM SINGLE SCAN

@author: tpisano
"""
import os, sys, collections, copy, shutil, numpy as np, scipy
from skimage.external import tifffile
from tools.utils.io import makedir, listdirfull, save_kwargs, listall
import multiprocessing as mp
from tools.imageprocessing.preprocessing import regex_determiner

if __name__=='__main__':
    
    

    src = '/home/wanglab/LightSheetTransfer/tp/180309_20170207_db_bl6_crii_rlat_03_647_010na_1hfds_z7d5um_100msec_10povlp_20-50-45_COPY'
    dst ='/home/wanglab/LightSheetTransfer/test_stitch2'
    dct=terastitcher_single_wrapper(src, dst, raw=True, regex=False, percent_overlap=0.1, create_image_dictionary=False)




#%%
def terastitcher_single_wrapper(src, dst, create_image_dictionary=False, **kwargs):
    '''Functions to handle folder consisting of files, stitch, resample, and combine.
    
    Inputs
    --------------
    src: folder of files
    dst: location to save - "fullsizedatafolder
    raw: if looking for raw data (LVBT - "raw_Datastack" in filename)
    regex: regular expression used. Must contain z,y,x,ch information (e.g. "(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ls>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.*r)(?P<ch>[0-9]{1,4})(.ome.tif)")
    voxel_size (tuple) of X,Y,Z dimensions. (e.g. (1.25, 1.25, 3.0))
    percent_overlap (flaot): percentage of overalp taken. (e.g. 0.1)

    NOT YET IMPLEMENTED
    #multipage: True: output single multipage tiff
               False: output each XY plane as a tiff
               

    #test 3 planes
    #image_dictionary['zchanneldct'] = {xx:image_dictionary['zchanneldct'][xx] for xx in ['0450', '0451', '0452']}
    '''
    #handle inputs:
    raw=kwargs['raw'] if 'raw' in kwargs else True
    regex=kwargs['regex'] if 'regex' in kwargs else False
    voxel_size=kwargs['voxel_size'] if 'voxel_size' in kwargs else (1.63, 1.63, 7.5)
    percent_overlap=kwargs['percent_overlap'] if 'percent_overlap' in kwargs else 0.1
    threshold=kwargs['threshold'] if 'threshold' in kwargs else 0.7
    algorithm=kwargs['algorithm'] if 'algorithm' in kwargs else 'MIPNCC'
    transfertype=kwargs['transfertype'] if 'transfertype' in kwargs else 'copy' #'move'
    outbitdepth = kwargs['outbitdepth'] if 'outbitdepth' in kwargs else 16
    cores = kwargs['cores'] if 'cores' in kwargs else 12
    verbose = kwargs['verbose'] if 'verbose' in kwargs else True
    resample = kwargs['resample'] if 'resample' in kwargs else False
    
    
    #handle file structure     
    if not regex: regex=regex_determiner(raw, src)
    sys.stdout.write('\nUsing regex as {}\n'.format(regex)); sys.stdout.flush()
    makedir(dst)
    
    #generate dictionary of files
    if create_image_dictionary: image_dictionary=create_image_dictionary(src, raw, regex)
    if not create_image_dictionary: image_dictionary=create_image_dictionary(src, raw, regex)
    
    
    #update
    dct = {'transfertype': transfertype, 'scalefactor':voxel_size, 'percent_overlap':percent_overlap, 'threshold':threshold, 'dst':dst, 'algorithm':algorithm, 'outbitdepth': outbitdepth}
    image_dictionary.update(dct)

    #determine jobs:
    jobdct=make_jobs(image_dictionary)
    
    #Terastitcher
    if cores>=2:
        #parallezation
        iterlst = [copy.deepcopy(inndct) for inndct in jobdct.values()]
        p = mp.Pool(cores)
        outlst = p.map(terastitcher_par, iterlst)
        p.terminate()

    else:
        outlst = [terastitcher_par(copy.deepcopy(inndct)) for inndct in jobdct.values()]
    
    #collapse        
    outdct = {xx[0]:[] for xx in outlst}; [outdct[xx[0]].append(xx[1]) for xx in outlst]
    
    #blend lighsheets
    if image_dictionary['lightsheets'] ==2: [blend_lightsheets(name, flds, dst, cores) for name, flds in outdct.iteritems()]
    if image_dictionary['lightsheets'] ==1: [blend_lightsheets(name, [flds[0], flds[0]], dst, cores) for name, flds in outdct.iteritems()] ##simulating two
    
    #save kwargs???
    
    return 



def blend_lightsheets(name, flds, dst, cores):
    '''
    '''
    sys.stdout.write('\nStarting blending of {}...'.format(name)); sys.stdout.flush()
    ydim, xdim =tifffile.imread(listall(flds[0], keyword='.tif')[0]).shape
    alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-250, 250, num=xdim)), (ydim, 1))
    fls0 = listall(flds[0], keyword='.tif'); fls0.sort()
    fls1 = listall(flds[1], keyword='.tif'); fls1.sort()
    assert set([os.path.basename(xx) for xx in fls0]) == set([os.path.basename(xx) for xx in fls1]), 'uneven number of z planes between L and R lightsheets'
    makedir(os.path.join(dst, name))
    iterlst=[{'alpha':alpha, 'fl0':fl0, 'fl1':fls1[i], 'dst':dst, 'name':name, 'zplane':i} for i,fl0 in enumerate(fls0)]
    if cores>=2:
        p=mp.Pool(cores)
        p.map(blend, iterlst)
        p.terminate()
    else:
        [blend(dct) for dct in iterlst]
    
    [shutil.rmtree(xx) for xx in flds]
    sys.stdout.write('completed.\n'.format(name)); sys.stdout.flush()
    return

def blend(dct):
    '''0=L, 1=R'''
    fl0 = dct['fl0']; fl1 = dct['fl1']
    alpha=dct['alpha']; im0 = tifffile.imread(fl0); dtype = im0.dtype; im1 = tifffile.imread(fl1)
    ch = '_C'+dct['fl0'][dct['fl0'].rfind('channel')+7:dct['fl0'].rfind('channel')+9]
    tifffile.imsave(os.path.join(dct['dst'], dct['name'], dct['name']+ch+'_Z'+str(dct['zplane']).zfill(4)+'.tif'), (im0*(1-alpha) + im1* (alpha)).astype(dtype), compress=1)
    [os.remove(xx) for xx in [fl0,fl1]]
    return


def terastitcher_par(inndct):
    '''Parallelize terastitcher using dct made by make_jobs function
    '''
    dct = inndct['dct']; out = inndct['out']; job=inndct['job']; channel = inndct['channel']; lightsheet = inndct['lightsheet']; name = inndct['name']
    transfertype = dct['transfertype']; voxel_size = dct['scalefactor']; percent_overlap = dct['percent_overlap']; dst=dct['dst']; algorithm = dct['algorithm']; outbitdepth=dct['outbitdepth']; threshold=dct['threshold']
    
    #format data
    make_folder_heirarchy(dct, dst=out, transfertype=transfertype, cores=1, scalefactor=voxel_size, percent_overlap=percent_overlap)    
        
    #stitch
    dst1 = os.path.join(dst, job+'_'+name)
    src = os.path.join(out, channel)
    call_terastitcher(src, dst1, voxel_size=voxel_size, threshold=threshold, algorithm = algorithm, outbitdepth = outbitdepth, resolutions='0', cleanup=True)
    
    return [name, dst1]


def make_jobs(image_dictionary):
    '''Simple function to create job dct for parallelization
    '''
    jobdct={}
    lslst = ['left_lightsheet', 'right_lightsheet']
    for channel in image_dictionary['channels']:
        for lightsheet in range(image_dictionary['lightsheets']):
            if image_dictionary['lightsheets'] == 2:
                side=['_C00_', '_C01_'][lightsheet]
                dct = copy.deepcopy(image_dictionary)
                dct['zchanneldct']={k:{kk:[xx for xx in vv if side in xx]} for k,v in dct['zchanneldct'].iteritems() for kk,vv in v.iteritems()}
                name = os.path.basename(os.path.dirname((dct['zchanneldct'].values()[0].values()[0][0])))
                out = os.path.join(dst, '{}_{}'.format(name, lslst[lightsheet])); makedir(out)
                jobdct['channel{}_lightsheet{}'.format(channel, lightsheet)] = copy.deepcopy({'job': 'channel{}_lightsheet{}'.format(channel, lightsheet), 'name': name, 'lightsheet': lslst[lightsheet], 'channel': channel, 'dst':dct['dst'], 'dct': copy.deepcopy(dct), 'out':out, 'cores':1})
    return jobdct


def call_terastitcher(src, dst, voxel_size=(1,1,1), threshold=0.7, algorithm = 'MIPNCC', outbitdepth = '16', resolutions='0', cleanup=True):
    '''
    Wrapper to use Terastitcher: https://doi.org/10.1186/1471-2105-13-316
    NOTE: terastitcher needs to be compiled using cmake and its path must be made global
        (usually done in bashrc): export PATH="/home/wanglab/TeraStitcher/src/bin:$PATH"

    
    Inputs
    -------------
    src = location of folder heirarchically formatted. See: make_folder_heirarchy
    dst = location to output to
    voxel_size = XYZ um/pixel
    threshold = https://github.com/abria/TeraStitcher/wiki/User-Interface#--thresholdreal
    algorithm = https://github.com/abria/TeraStitcher/wiki/User-Interface#--algorithmstring-advanced
    outbitdepth = 8, 16, .... https://github.com/abria/TeraStitcher/wiki/User-Interface#--imout_depthstring
    resolutions = 0, 01, 012, ... https://github.com/abria/TeraStitcher/wiki/User-Interface#--resolutionsstring
    cleanup = remove files after stitching
    
    Returns:
    folder location
    
    command line example (inpsired by: https://github.com/abria/TeraStitcher/wiki/Demo-and-batch-scripts)
    terastitcher --import --volin=/home/wanglab/LightSheetTransfer/test_stitch/00 --volin_plugin="TiledXY|3Dseries" --imin_plugin="tiff3D" --imout_plugin="tiff3D" --ref1=1 --ref2=2 --ref3=3 --vxl1=1 --vxl2=1 --vxl3=1 --projout=xml_import
    terastitcher --displcompute --projin='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_import.xml'
    terastitcher --displproj --projin='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_import.xml'
    terastitcher --displthres --projin='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displproj.xml' --projout='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displthres.xml' --threshold=0.7
    terastitcher --placetiles --projin='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displthres.xml' --projout='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_placetiles.xml' --algorithm=MIPNCC
    terastitcher --merge --projin='/home/wanglab/LightSheetTransfer/test_stitch/00/xml_placetiles.xml' --volout='/home/wanglab/LightSheetTransfer/test_stitch/00' --imout_depth=16 --resolutions=012345
    
    '''
    import subprocess as sp
    #import
    xml_import = os.path.join(src, 'xml_import.xml')
    call0 = 'terastitcher --import --volin={} --volin_plugin="TiledXY|3Dseries" --imin_plugin="tiff3D" --imout_plugin="tiff3D" --ref1=1 --ref2=2 --ref3=3 --vxl1={} --vxl2={} --vxl3={} --projout={}'.format(src, voxel_size[0],voxel_size[1], voxel_size[2], xml_import)
    sp_call(call0)
    
    #align
    sys.stdout.write('\n\nRunning Terastitcher alignment on {}, this can take some time....'.format(' '.join(src.split('/')[-2:]))); sys.stdout.flush()
    xml_displcomp = os.path.join(src, 'xml_displcomp.xml')
    call1 = "terastitcher --displcompute --projin={} --projout={}".format(xml_import, xml_displcomp)
    sp_call(call1)
    sys.stdout.write('completed!'); sys.stdout.flush()
    
    #projection
    xml_displproj = os.path.join(src, 'xml_displproj.xml')
    call2 = "terastitcher --displproj --projin={}".format(xml_import)
    sp_call(call2)
    
    #threshold
    xml_displthresh = os.path.join(src, 'xml_displthresh.xml')
    call3 = "terastitcher --displthres --projin={} --projout={} --threshold={}".format(xml_displproj, xml_displthresh, threshold)
    sp_call(call3)
    
    #placetiles
    xml_placetiles = os.path.join(src, 'xml_placetiles.xml')
    call4 = "terastitcher --placetiles --projin={} --projout={} --algorithm={}".format(xml_displthresh, xml_placetiles, algorithm)
    sp_call(call4)
    
    #merge
    sys.stdout.write('\nOutputting images, this can also take some time....'); sys.stdout.flush()
    makedir(dst)
    call5 = "terastitcher --merge --projin={} --volout={} --imout_depth={} --resolutions={}".format(xml_placetiles, dst, outbitdepth, resolutions)
    sp_call(call5)
    sys.stdout.write('completed!'); sys.stdout.flush()
    
    #cleanup
    if cleanup: 
        sys.stdout.write('\nCleaning up....'); sys.stdout.flush()
        shutil.rmtree(os.path.dirname(src))
        sys.stdout.write('completed! :]'); sys.stdout.flush()    
    
    #folder containing list of tiffs
    return #listdirfull([xx for xx in listdirfull(listdirfull(dst)[0]) if os.path.isdir(xx)][0])[0]

def sp_call(call):
    #p = sp.Popen(call, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    #(output,err) = p.communicate()
    #p_status = p.wait()
    #print out
    #print err
    #print p_status 
    '''
    import subprocess, sys
    p = subprocess.Popen(call, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        out = p.stderr.read(1)
        if out == '' and p.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()'''
    from subprocess import PIPE, Popen
    return Popen(call, stdout=PIPE, shell=True).stdout.read()
    
    
#%% 
def make_folder_heirarchy(image_dictionary, dst=False, transfertype='move', scalefactor=(1.63, 1.63, 7.5), percent_overlap=0.1, cores=False, **kwargs):
    '''Function to make folders for compatibility with Terastitcher
    
    Inputs:
    --------------
    image_dictionary: dctionary generated from make_image_dictionary
    dst (optional): to make folder structure somewhere else
    transfertype (optional): 'move': move files from current location to dst
                             'copy': copy files from current location to dst
                                     
    Returns:
    -----------------
    paths to each channel folder
    '''    
    #inputs
    if not dst: dst = image_dictionary['sourcefolder']
    makedir(dst)
    
    #image dims
    ypx,xpx = tifffile.imread(listdirfull(image_dictionary['sourcefolder'])[0]).shape
    
    #factor in percent overlap
    ypx = ypx * (1-percent_overlap)
    xpx = xpx * (1-percent_overlap)
    
    #tiles
    xtile = image_dictionary['xtile']
    ytile = image_dictionary['ytile']
    
    sys.stdout.write('\nMaking Folders,'); sys.stdout.flush()    
    
    #'''WORKED BUT NEED TO FLIP Z AND Y
    iterlst = []
    for ch in image_dictionary['channels']:
        chdst = dst+'/'+ch; makedir(chdst)
        for y in range(image_dictionary['ytile']):
            ynm = str(int(ypx*y*scalefactor[1])*10).zfill(6)
            ydst = chdst+'/'+ynm; makedir(ydst)
            for x in range(image_dictionary['xtile']):
                xnm = str(int(xpx*x*scalefactor[0])*10).zfill(6)
                xdst = ydst+'/'+ynm+'_'+xnm; makedir(xdst)
                for z in image_dictionary['zchanneldct']:
                    znm = str(int(int(z)*scalefactor[2])*10).zfill(6)
                    lst = image_dictionary['zchanneldct'][str(z).zfill(4)][ch]; lst.sort()
                    iterlst.append((lst[(y*(ytile)+x)], xdst+'/'+ynm+'_'+xnm+'_'+znm+'.tif', transfertype))
                    #print y,x,z,znm, (y*(ytile)+x), ynm, xnm, znm, os.path.basename(lst[(y*(ytile)+x)])[20:60]
                    
    #generate backup just in case
    #try:
    #    kwargs['terastitcher_dct'] = {xx[0]:xx[1] for xx in iterlst}
    #    save_kwargs(**kwargs)
    #except Exception, e:
    #    print ('Exception: {}...not saving terastitcher_dct'.format(e))
                    
    #move/copy files
    if cores>=2:
        sys.stdout.write(' populating folders: {} files using {} cores...\n'.format(len(iterlst), cores)); sys.stdout.flush()
        p=mp.Pool(cores)
        p.map(make_folder_heirarchy_helper, iterlst)
        p.terminate()
        
    else:
        sys.stdout.write(' populating folders...'); sys.stdout.flush()
        [make_folder_heirarchy_helper(i) for i in iterlst]          
        
    sys.stdout.write('finished.\n'); sys.stdout.flush()        
    return
    
def make_folder_heirarchy_helper((src, dst, transfertype)):
    '''
    '''
    import shutil
    if transfertype == 'move': shutil.move(src, dst)
    elif transfertype == 'copy': shutil.copy(src, dst)
    
    return
    

    
def find_tiff_dims(src):
    '''Find dimensions of a tifffile without having to load
    
    Inputs
    --------
    src: file path
    
    Returns
    --------
    pages: number of pages in tiff, for LVBT this is the number of horizontal foci
    y,x: pixel dims
    '''
    from skimage.external import tifffile
    with tifffile.TiffFile(os.path.join(src)) as tif:
        pages = len(tif.pages) #number of horizontal foci
        y, x = tif.pages[0].shape
        tif.close()
    
    return pages, y, x


    
    #%%
def flatten_stitcher(cores, outdr, ovlp, xtile, ytile, zpln, dct, lightsheets, blndtype = False, intensitycorrection = False, **kwargs):
    '''return numpy arrays of     '''
    ###zpln='0500'; dct=zdct[zpln]    
    #easy way to set ch and zplnlst   
    ['stitching for ch_{}'.format(ch[-2:]) for ch, zplnlst in dct.iteritems()] #cheating way to set ch and zplnlst 
    ###dim setup   
    ydim, xdim = cv2.imread(zplnlst[0]).shape    
    xpxovlp = int(ovlp*xdim)
    ypxovlp = int(ovlp*ydim)
    tiles = len(zplnlst) ##number of tiles
    
    if not blndtype: blndtype = kwargs['blendtype']
    if not intensitycorrection: intensitycorrection = kwargs['intensitycorrection']
    
    #check for optional blendingfactor:
    if 'blendfactor' in kwargs:
        blendfactor = int(kwargs['blendfactor'])
    else:
        blendfactor = 4
    
### blending setup
    if blndtype == 'linear':        
        alpha = np.tile(np.linspace(0, 1, num=xpxovlp), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(np.linspace(0, 1, num=ypxovlp), ((xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)
    elif blndtype == 'sigmoidal':
        alpha = np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=xpxovlp)), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=ypxovlp)), ((xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)
    elif blndtype == False or blndtype == None: #No blending: generate np array with 0 for half of overlap and 1 for other. 
        alpha = np.zeros((ydim, xpxovlp)); alpha[:, xpxovlp/2:] = 1
        yalpha = np.zeros((ypxovlp, (xdim+((1-ovlp)*xdim*(xtile-1))))); yalpha[ypxovlp/2:, :] = 1
    else:
        alpha = np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=xpxovlp)), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=ypxovlp)), ((xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)
        
##parallel processing             
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #stitchlst=Parallel(n_jobs=cores)(delayed(flatten_xystitcher)(dr, xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch[-2:], zplnlst) for ch, zplnlst in dct.iteritems())
        try:
            p
        except NameError:
            p=mp.Pool(cores)
        iterlst=[]; [iterlst.append((xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch[-2:], zplnlst, intensitycorrection, lightsheets)) for ch, zplnlst in dct.iteritems()]           
        stitchlst=p.map(flatten_xystitcher, iterlst)
        stitchdct={}; [stitchdct.update(i) for i in stitchlst]
    del ydim, xdim, xpxovlp, ypxovlp, tiles, alpha, yalpha, iterlst, stitchlst
    p.terminate()
    return stitchdct

def create_image_dictionary(src, raw, regex):
    '''Regex to create zplane dictionary of subdictionaries (keys=channels, values=single zpln list sorted by x and y).
    USED FOR RAW DATA FROM LVBT: 1-2 light sheets (each multipage tiff, where a page represents a horizontal foci)'''    
    import re
    #find files
    if raw: fl=[os.path.join(src, f) for f in os.listdir(src) if 'raw_DataStack' in f] #sorted for raw files
    if not raw: fl=[os.path.join(src, f) for f in os.listdir(src) if not 'raw_DataStack' in f] #sorted for raw files
    reg=re.compile(regex)
    matches=map(reg.match, fl) #matches[0].groups()
    
    #find index of z,y,x,ch in a match str
    z_indx=matches[0].span('z')
    try:    
        y_indx=matches[0].span('y')    
        x_indx=matches[0].span('x')
        tiling = True
    except IndexError:        
        y_indx = 1
        x_indx = 1
        tiling = False
    
    #determine number of channels, sheets, horizontal foci
    chs=[]; [chs.append(matches[i].group('ch')[:]) for i in range(len(matches)) if matches[i].group('ch')[:] not in chs]
    zplns=[]; [zplns.append(matches[i].group('z')) for i in range(len(matches)) if matches[i].group('z') not in zplns]; zplns.sort()
    
    #find dims
    hf, y, x = find_tiff_dims(os.path.join(src, ''.join(matches[0].groups())))
    
    #make dct consisting of each channel sorted by z plane, then in xy order (topleft-->top right to bottom-right), then sorted for ls(L then R)
    zdct={}; chdct={}; bd_dct={} #check for bad planes (represented in one channel but not the other)
    for ch in chs:      
        #num of chs
        lst=[]; [lst.append(''.join(match.groups())) for num,match in enumerate(matches) if ch in match.group('ch')]
        try:        
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then ls, then x, then y
        except NameError:
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then x, then y            
        if tiling:
            ytile=int(max([yy for f in lst for yy in f[y_indx[0]:y_indx[1]]]))+1 #automatically find the number of tiles
            xtile=int(max([xx for f in lst for xx in f[x_indx[0]:x_indx[1]]]))+1 #automatically find the number of tiles  
        elif not tiling:
            ytile = 1 
            xtile = 1
        try:
            ls_indx=matches[0].span('ls')
            lsheets=[]; [lsheets.append(matches[i].group('ls')[-2:]) for i in range(len(matches)) if matches[i].group('ls')[-2:] not in lsheets]
            lsheets=int(max([lsh for f in lst for lsh in f[ls_indx[0]:ls_indx[1]]]))+1 #automatically find the number of light sheets  
            intvl=xtile*ytile*lsheets
        except IndexError:
            intvl=xtile*ytile
            lsheets=1
        ################find z plns missing tiles and pln to badlst
        test_matches=map(reg.match, srtd)
        new_z_indx=test_matches[0].span('z')
        z_lst=[xx[new_z_indx[0]:new_z_indx[1]] for xx in srtd]
        counter=collections.Counter(z_lst)
        bd_dct[ch]=[xx for xx in counter if counter[xx] != intvl]
        ############sort by plane
        ttdct={}
        for plns in zplns:
            try:            
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))
            except NameError:
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))                    
            ttdct[plns]=tmp
        ########key=channel; values=dictionary of tiles/pln
        chdct[ch[-2:]]=ttdct
    ###max zpln
    mx_zpln=max([len(chdct[xx]) for xx in chdct])        
    ###zdct: keys=pln, values=dictionary of channels with subvalues being tiles/lightsheet
    for xx in range(mx_zpln-1):        
        tmpdct={}
        [tmpdct.update(dict([(chann, chdct[chann][str(xx).zfill(4)])])) for chann in chdct]
        zdct[str(xx).zfill(4)]=tmpdct
    ################################################################################################
    ###REMOVE ENTIRE PLANE, ALL CHANNEL WHERE THERE IS MISSING FILES; THIS MIGHT NEED TO BE REVISITED
    for chann in bd_dct:
        if len(bd_dct[chann]) > 0:
            for bdpln in bd_dct[chann]:            
                del zdct[bdpln]
    ################################################################################################    
    chs=[ch[-2:] for ch in chs] 
    ###check to see if all channels have the same length, if not it means LVBT messed up
    if max([len(bd_dct[xxx]) for xxx in bd_dct]) >0:
        print ('Unequal_number_of_planes_per_channel_detected...seriously WTF LVBT.\n')
        print ('\nChannels and planes that were bad {}'.format(bd_dct))
        print ('\nBad planes have been removed from ALL CHANNELS')
    #####find full size dimensions in zyx
    print("{} *Complete* Zplanes found for {}\n".format(len(zdct.keys()), src))
    print('Checking for bad missing files:\n     Bad planes per channel:\n     {}\n'.format(bd_dct))
    print("{} Channels found\n".format(len(zdct['0000'].keys())))
    print("{}x by {}y tile scan determined\n".format(xtile, ytile))           
    print("{} Light Sheet(s) found. {} Horizontal Focus Determined\n\n".format(lsheets, hf))
    return dict([('zchanneldct', zdct), ('xtile', xtile), ('ytile', ytile), ('channels', chs), ('lightsheets', lsheets), ('horizontalfoci', hf), ('fullsizedimensions', (len(zdct.keys()),(y*ytile),(x*xtile))), ('sourcefolder', src)])