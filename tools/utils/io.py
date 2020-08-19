#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:14:00 2017

@author: wanglab
"""
from math import ceil
import multiprocessing as mp
import numpy as np
#import dill as pickle
import pickle
import os, sys, time, shutil
import tifffile
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import zoom
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.parallel import parallel_process



def makedir(path):
    '''Simple function to make directory if path does not exists'''
    if os.path.exists(path) == False:
        os.mkdir(path)
    return

def removedir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    return

def listdirfull(x, keyword=False):
    '''might need to modify based on server...i.e. if automatically saving a file called 'thumbs'
    '''
    if not keyword:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx]
    else:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx and keyword in xx]

def chunkit(core, cores, item_to_chunk):
    '''function used for parallel processes to determine the chunk range they should process, returns tuple of lower and upper ranges
    assumes zero indexing for the "core" input
    '''

    if type(item_to_chunk)==int:
        item_to_chunk=range(item_to_chunk)
    chnksz=int(ceil(len(item_to_chunk)/(cores)))
    ###if single core
    if cores == 1:
        chnkrng=(0, chnksz)
    elif core == 0:
        chnkrng=(chnksz*(core), chnksz*(core+1)-1)
    elif core != cores-1 and core != 0:
        chnkrng=(chnksz*(core)-1, chnksz*(core+1)-1)
    elif core == cores-1:
        chnkrng=(chnksz*(core)-1, len(item_to_chunk)) #remainder for noneven chunking
    return chnkrng


def writer(saveloc, texttowrite, flnm=None, verbose=True):
    '''Function to write string of text into file title FileLog.txt.
    Optional flnm input to change name of log'''
    if flnm == None:
        flnm = "LogFile.txt"
    if verbose==True:
        if os.path.exists(saveloc) == False:
            with open(os.path.join(saveloc, flnm), "w") as filelog:
                filelog.write(texttowrite)
                filelog.close()
            
            return
        elif os.path.exists(saveloc) == True:
            with open(os.path.join(saveloc, flnm), "a") as filelog:
                filelog.write(texttowrite)
                filelog.close()
            print(texttowrite)
            return
        else:
            print ('Error using tracer.writer function')
            return
    elif verbose==False:
        if os.path.exists(saveloc) == False:
            with open(os.path.join(saveloc, flnm), "w") as filelog:
                filelog.write(texttowrite)
                filelog.close()
            return
        elif os.path.exists(saveloc) == True:
            with open(os.path.join(saveloc, flnm), "a") as filelog:
                filelog.write(texttowrite)
                filelog.close()
            return
        else:
            print ('Error using tracer.writer function')
            return

def get_filepaths(directory):
    """
    #from: https://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python/19308592#19308592?newreg=eda6052e44534b0982fb01506d1a2fbf

    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def dict_update(pth, old_system_directory = False):
    '''Function to update param dictionary by replacing both text of system direcotry 
    with new AND replace kwargs['systemdirectory']

    pth = '/home/wanglab/Downloads/param_dict.p'
    old_system_directory (optional) - str or list to search for
    #homemade hack to remove other systemdirectories
    for xx in lst:
        kwargs=load_kwargs(xx)
        kwargs['systemdirectory'] = '/mnt/bucket/labs/'
        from tools.utils.io import save_kwargs
        save_kwargs(**kwargs)
        kwargs=load_kwargs(xx)
        updateparams(**kwargs)

    '''
    kwargs = {}
    try:
        with open(pth, 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()
    except IOError:
        pth = pth+'/param_dict.p'
        with open(pth, 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()

    if not old_system_directory:
        old_system_directory = [kwargs['systemdirectory']]
    elif type(old_system_directory) == str: old_system_directory = [old_system_directory]

    old_system_directory.append(kwargs['systemdirectory'])

    old_system_directory = list(set(old_system_directory))

    new_system_directory = directorydeterminer()

    try:
        old_system_directory.remove(new_system_directory)
    except:
        pass

    return

def load_kwargs(outdr=None, update_dict = True, system_directories = ['/jukebox/', '/mnt/bucket/labs/', '/home/wanglab/', '/home/tpisano'], **kwargs):
    '''simple function to load kwargs given an 'outdr'

    Inputs:
    -------------
    outdr: (optional) path to folder generated by package OR path to param dictionary
    kwargs
    '''
    #handle optional inputs
    if outdr: kwargs = {}; kwargs = dict([('outputdirectory',outdr)])

    #optionally update dictionary
    if update_dict: dict_update(kwargs['outputdirectory'], old_system_directory = system_directories)

    try:
        with open(os.path.join(kwargs['outputdirectory'], 'param_dict.p'), 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()
    except IOError:
        with open(kwargs['outputdirectory'], 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()

    return kwargs

def save_kwargs(pckloc = None, verbose = False, **kwargs):
    '''Save out kwargs as param_dict.p unless otherwise noted.

    Inputs
    ----------------
    pckloc (optional) = location and name to save .p file
    kwargs = dictionary to pickle

    Returns
    ----------------
    location/name of file
    '''
    #handle input
    if not pckloc: pckloc=os.path.join(kwargs['outputdirectory'], 'param_dict.p')

    #verbosity
    if verbose: sys.stdout.write('\n\npckloc: {}'.format(pckloc))

    #pickle
    pckfl=open(pckloc, 'wb'); pickle.dump(kwargs, pckfl); pckfl.close()

    return pckloc

def save_dictionary(dst, dct):
    '''Basically the same as save_kwargs
    '''
    import cPickle as pickle
    if dst[-2:]!='.p': dst=dst+'.p'
    
    with open(dst, 'wb') as fl:    
        pickle.dump(dct, fl, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_dictionary(pth):
    '''simple function to load dictionary given a pth
    '''
    kwargs = {};
    with open(pth, 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs

def load_tif_list(inn, cores = None, save = False):
    '''Function to load a list of tiff files into a single stack. Python's Tifffile package has this feature but for unknown reasons causes memory issues on the cluster. A cheap work around.

    Inputs:
    --------------------
    inn:
        list of files to load
        --or--
        pth to folder of files to load
    cores (optional, only get a performance boost if saving):
        None: do not parallelize, this is safer for more memory intensive files
        ##: number of cores to allocate for this. Suggested to do at least one core less than available.

    Returns:
    -------------------
    save (optional):
        False: returns: np array of loaded files
        str: path to save numpy memory mapped file

    '''
    #handle inputs
    assert type(inn) == list or type(inn) == str, 'inn must be a string or lst'
    if cores: assert save, 'For parallelization to be efficient, you must save the memory mapped file, set save path'

    #if pth to folder
    if type(inn) == str: inn = [xx for xx in listdirfull(inn) if 'tif' in xx and '~' not in xx]; inn.sort()
    dims = [len(inn)]+list(tifffile.imread(inn[0], multifile = False).shape)

    if not cores or cores==1:
        #allocate memory:
        imstack = np.zeros(dims)

        for i in range(imstack.shape[0]):
            imstack[i,...] = tifffile.imread(inn[i], multifile = False)

        #returns
        if save:
            if not save[-4:] == '.tif': save = save + '.tif'
            tifffile.imsave(save, imstack)
        else:
            return imstack


    #parallelized:
    else:

        #memory mapped file
        if not save[-4:] == '.npy': save = save + '.npy'

        arr = load_memmap_arr(save, dtype = 'uint16', mode = 'w+', shape = tuple(dims)); del arr

        #paralellize
        p = mp.Pool(cores)
        iterlst = []; [iterlst.append((core, cores, inn, save)) for core in range(cores)]
        p.starmap(load_tif_helper, iterlst)
        p.terminate()

        #return
    return save



######################## helper functions for parallelization ############################
def load_tif_helper(core, cores, inn, save):
    '''helper function for parallelization of image loading
    inputs:
        core/cores = # of # processes
        inn = list
        fp = np memory mapped file

    '''
    #handle subprocesses
    chnkrng=chunkit(core, cores, inn)
    print('   chunk {} of {} - planes {}-{}\n'.format((core+1), (cores), chnkrng[0], chnkrng[1]))

    arr = np.lib.format.open_memmap(save, dtype = 'uint16', mode = 'r+')

    #load and save to memory mapped file
    #arr[chnkrng[0]:chnkrng[1],...] = np.squeeze(tifffile.imread(inn[chnkrng[0]:chnkrng[1]]))
    for i in range(chnkrng[0], chnkrng[1]):
        arr[i,...] = tifffile.imread(inn[i], multifile=False)
        arr.flush()
    del arr

    return


def compress_full_sizedatafld(jobid, cores=None, chunksize=None, compression=1, removeoriginal=False, **kwargs):
    '''Function to compress tiffs for longer term storage
    Inputs:
        compresssion: 0-9
        removeoriginal: if true will delete file from fullsizeddatafolder
    '''
    ###########Inputs###############
    outdr = kwargs['outputdirectory']
    pckl = open(os.path.join(outdr, 'param_dict.p'), 'rb'); kwargs.update(pickle.load(pckl)); pckl.close()
    full_sizedatafld = kwargs['full_sizedatafld']
    chs=kwargs['channels']
    zplns=kwargs['fullsizedimensions'][0]
    if chunksize == None:
        chunksize = zplns
    if cores == None:
        cores = 5
    ########Multiprocessing##########
    try:
        p
    except NameError:
        p = mp.Pool(cores)
    print ('Jobid {} running with {} cores'.format(jobid, cores))

    ###find files
    fls = [os.path.join(xx, yy) for xx in [os.path.join(full_sizedatafld, 'ch{}'.format(x)) for x in chs] for yy in os.listdir(xx)]; fls.sort()

    ###parse up jobs
    if jobid > ceil(len(fls) / chunksize):
        print ('job not needed; jobid({}) > fls/chunksize({})'.format(jobid, ceil(len(fls) / chunksize)))
        return

    ###each job's coverage.
    fls_to_process = fls[(jobid*chunksize): ((1+jobid)*chunksize)]

    ###run jobs
    start = time.time()
    iterlst = []; [iterlst.append((jobid, fls_to_process, compression, removeoriginal, core, cores)) for core in range(cores)]
    lst = p.starmap(compress_full_helper, iterlst); lst.sort()
    print ('Completed {} through {}\nUsing {} cores'.format(fls_to_process[0], fls_to_process[-1], time.time() - start))

def compress_full_helper(jobid, fls_to_process, compression, removeoriginal, core, cores):
    '''helper function to compress tifs
    '''
    #from tools.imageprocessing.preprocessing import chunkit
    chnkrng=chunkit(core, cores, fls_to_process)
    for i in range(chnkrng[0], chnkrng[1]):
        fl=fls_to_process[i]
        pth=fl[:fl.rfind('/')]+'_compressed'
        print(pth)
        makedir(pth)
        svnm=os.path.join(pth, fl[fl.rfind('/')+1:])
        tifffile.imsave(svnm, tifffile.imread(fl), compress=compression)
        print(svnm)
        if removeoriginal==True:
            del fl
    return

def uncompress(src, dst=False, cores = None):
    '''function to uncompress tiffs

    Inputs:
    -----------------------
    src = location of compressed tiffs
    dst = (optional) specify for different output -
    cores (optional) = number of cores to parallelize with. If None, defaults to machine's cores - 1

    '''

    if not cores: cores = int(mp.cpu_count() - 1)
    if dst: makedir(dst)

    if cores == 1:
        if not dst: [tifffile.imsave(fl, tifffile.imread(fl)) for fl in listdirfull(src) if fl[-4:] == '.tif']
        if dst: [tifffile.imsave(os.path.join(dst, fl[fl.rfind('/')+1:]), tifffile.imread(fl)) for fl in listdirfull(src) if fl[-4:] == '.tif']

    else:
        sys.stdout.write('\n\nUncompressing with {} cores:\n'.format(cores)); sys.stdout.flush()
        jobs = listdirfull(src)
        joblength = range(len(jobs))
        iterlst=[]; [iterlst.append((jobs[jobid], 0, dst)) for jobid in joblength]
        parallel_process(iterlst, compress_helper, n_jobs=cores)


    return

def compress(src, compression=1, dst=False, cores = None):
    '''function to compress tiffs within a folder

    Inputs:
    -----------------------
    src = location of compressed tiffs
    dst = (optional) specify for different output
    compression = level of compression 0(none)-9(max)
    cores (optional) = number of cores to parallelize with. If None, defaults to machine's cores - 1

    '''
    if not cores: cores = int(mp.cpu_count() - 1)
    if dst: makedir(dst)

    if cores == 1:
        if not dst: [tifffile.imsave(fl, tifffile.imread(fl), compress=compression) for fl in listdirfull(src) if fl[-4:] == '.tif']
        if dst: [tifffile.imsave(os.path.join(dst, fl[fl.rfind('/')+1:], compress=compression), tifffile.imread(fl)) for fl in listdirfull(src) if fl[-4:] == '.tif']
    else:
        #paralellize
        sys.stdout.write('\n\nCompressing with {} cores. '.format(cores)); sys.stdout.flush()
        jobs = listdirfull(src)
        joblength = range(len(jobs))
        iterlst=[]; [iterlst.append((jobs[jobid], compression, dst)) for jobid in joblength]
        sys.stdout.write('{} files to compress...\n'.format(len(jobs))); sys.stdout.flush()
        parallel_process(iterlst, compress_helper, n_jobs=cores)

    return

def compress_helper(fl, compression, dst):
    '''helper function for parallelization of compression
    '''
    #determine if dst is different from src
    if not dst:

        #process with same dst
        if fl[-4:] == '.tif': tifffile.imsave(fl, tifffile.imread(fl), compress=compression)

    #different dst
    elif dst:
        if fl[-4:] == '.tif': tifffile.imsave(os.path.join(dst, fl[fl.rfind('/')+1:]), tifffile.imread(fl), compress = compression)

    return


def listall(fld, keyword=False):
    '''function to recursively list all files within a folder and subfolders

    Inputs:
    -----------------------
    fld = folder to search

    '''
    fls = []
    for (root, dir, files) in os.walk(fld):
         for f in files:
             path = os.path.join(root, f)
             if os.path.exists(path):
                 fls.append(path)
                 
    if keyword: fls = [xx for xx in fls if keyword in xx]
    return fls



def replaceword(fl, oldword, replacement):

    assert type(oldword) == str
    assert type(replacement) == str

    with open(fl,'r') as f:
        filedata = f.read()
        f.close()

    newdata = filedata.replace(oldword, replacement)

    with open(fl,'w') as f:
        f.write(newdata)
        f.close()

    return


def resize(src, dst, zoomfactor=False, compression=1, cores = None):
    '''Function to take as input folder of tiff images, resize and save

    Inputs
    ---------------------
    src = input folder of tiffs
    dst = output folder to save
    zoomfactor = (optional) float to scale image by. smaller<1<larger
    compression = (optional) compression factor
    cores = number of parallel processses
    '''

    if not cores: cores = mp.cpu_count() - 1
    if dst[-1] == '/': dst = dst[:-1]
    makedir(dst)
    if not zoomfactor: zoomfactor = 1

    sys.stdout.write('\n\nCompressing with {} cores. '.format(cores)); sys.stdout.flush()
    jobs = listdirfull(src)
    iterlst=[]; [iterlst.append((fl, dst, zoomfactor, compression)) for fl in listdirfull(src) if '.tif' in fl and '~' not in fl and 'Thumbs.db' not in fl]
    sys.stdout.write('{} files to compress...\n'.format(len(jobs))); sys.stdout.flush()
    parallel_process(iterlst, resize_helper, n_jobs=cores)
    sys.stdout.write('...done'.format(len(jobs))); sys.stdout.flush()
    return


def resize_helper(fl, dst, zoomfactor, compression):
    '''
    '''
    return tifffile.imsave(dst+fl[fl.rfind('/'):], zoom(tifffile.imread(fl, series=0, key=0), zoomfactor), compress = compression)

def resize_list(src, dst, zoomfactor=False, compression=1, cores = None):
    '''Function to take a list of tiff images as input, resize and save

    Inputs
    ---------------------
    src = input LIST of tiffs
    dst = output folder to save
    zoomfactor = (optional) float to scale image by. smaller<1<larger
    compression = (optional) compression factor
    cores = number of parallel processses
    '''

    if not cores: cores = mp.cpu_count() - 1
    if dst[-1] == '/': dst = dst[:-1]
    makedir(dst)
    if not zoomfactor: zoomfactor = 1

    sys.stdout.write('\n\nCompressing with {} cores. '.format(cores)); sys.stdout.flush()
    iterlst=[]; [iterlst.append((fl, dst, zoomfactor, compression)) for fl in src if '.tif' in fl and '~' not in fl and 'Thumbs.db' not in fl]
    sys.stdout.write('{} files to compress...\n'.format(len(src))); sys.stdout.flush()
    parallel_process(iterlst, resize_helper, n_jobs=cores)
    sys.stdout.write('...done'.format(len(src))); sys.stdout.flush()
    return

def change_bitdepth(src, outdepth = 'uint8'):
    '''Function to take numpy array, rescale_intensity then change bitdepth

    Inputs
    ---------------------
    src = numpy array
    outdepth = bitdepth to output

    '''
    assert type(outdepth) == str, 'outdepth variable must be a string'

    #return rescale_intensity(src, in_range=str(src.dtype), out_range = outdepth).astype(outdepth) #doesn't work as well
    return rescale_intensity(src, out_range = outdepth).astype(outdepth)

def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

def save_memmap_arr(pth, arr):
    '''Function to save memmaped array.
    
    pth = place to save, will overwrite
    arr = arr

    '''
    narr = np.lib.format.open_memmap(pth, dtype = arr.dtype, shape = arr.shape, mode = 'w+')
    narr[:] = arr
    narr.flush(); del narr
    return

def load_np(src, mode='r'):
    '''Function to handle .npy and .npyz files. Assumes only one k,v pair in npz file
    '''
    if str(type(src)) == "<type 'numpy.ndarray'>" or str(type(src)) == "<class 'numpy.core.memmap.memmap'>":
        return src
    elif src[-4:]=='.npz':
        fl = np.load(src)
        #unpack ASSUMES ONLY SINGLE FILE
        arr = [fl[xx] for xx in fl.keys()][0]
        return arr
    elif src[-4:]=='.npy':
        try:
            arr=load_memmap_arr(src, mode)
        except:
            arr = np.load(src)
        return arr

def shape_np(src):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with open(src, 'r') as f:
        s= f.readlines(1)
    s=s[0]
    ln = s[s.find('shape')+8:s.find('}')-2].replace('(','').replace(')','').split(',')
    return tuple([int(xx) for xx in ln])

def change_arr_given_indices_condition(arr, indices, condition, new_value):
    '''Function to change a subset of an array given indices, condition and new value
    
    inputs: 
        arr = numpy array
        indicies = tuple
        condition = condition when true to change
        new_value = value to change
        
    e.g. arr, indices = (254, slice(55,67), slice(100,125)), condition = 1064, new_value = 936
    changes the 
    '''
    mask = arr[indices]
    mask[np.where(mask == condition)] = new_value
    arr[indices] = mask
    return arr


def make_memmap_from_np_list(src, dst, cores = 1, verbose = True):
    '''Function to make a memory mapped array from a list of numpy files

    Concatenates along 0 axis
    '''
    #find dtype and dims
    dims = [shape_np(xx) for xx in src]
    shape = list(dims[0])
    shape[0] = sum([xx[0] for xx in dims])
    if verbose: sys.stdout.write('Generating memory mapped array of dims {}:'.format(tuple(shape))); sys.stdout.flush()

    #indices
    indx = 0
    indices = [0]
    for d in dims:
        indx+=d[0]
        indices.append(indx)

    if cores <= 1 or cores == False:
        #init
        for i in range(len(src)):
            if verbose: sys.stdout.write('\n   loading {} of {}'.format(i, len(src))); sys.stdout.flush()
            arr = np.load(src[i])

            #init
            if i == 0: memmap=load_memmap_arr(dst, mode='w+', dtype=arr.dtype, shape=tuple(shape))

            #
            memmap[indices[i]:indices[i+1],...] = arr
            if verbose: sys.stdout.write('...flushing to disk.'); sys.stdout.flush()
            memmap.flush()
    else:
        #init
        if verbose: sys.stdout.write('\n   loading {} of {}'.format(0, len(src))); sys.stdout.flush()
        arr = np.load(src[0])

        #init
        memmap=load_memmap_arr(dst, mode='w+', dtype=arr.dtype, shape=tuple(shape))
        memmap[indices[0]:indices[1],...] = arr
        if verbose: sys.stdout.write('...flushing to disk.'); sys.stdout.flush()

        #parallelize rest:
        p = mp.Pool(cores)
        iterlst = [(i, indices, shape, src, dst, verbose) for i in range(1, len(src))]
        p.map(make_memmap_from_np_list_helper, iterlst)

    return dst

def make_memmap_from_np_list_helper(i, indices, shape, src, dst, verbose):
    '''
    '''
    if verbose: sys.stdout.write('\n   loading {} of {}...'.format(i, len(src))); sys.stdout.flush()
    arr = np.load(src[i])
    memmap=load_memmap_arr(dst, mode='r+', dtype=arr.dtype, shape=tuple(shape))
    if verbose: sys.stdout.write('\n...flushing {} to disk.'.format(i)); sys.stdout.flush()
    memmap[indices[i]:indices[i+1],...] = arr
    memmap.flush()
    del memmap
    return

def make_memmap_from_tiff_list(src, dst, dtype=False):
    '''Function to make a memory mapped array from a list of tiffs
    '''
    if type(src) == str and os.path.isdir(src): 
        src = listdirfull(src, keyword = '.tif')
        src.sort()
    im = tifffile.imread(src[0])
    if not dtype: dtype = im.dtype
    memmap=load_memmap_arr(dst, mode='w+', dtype=dtype, shape=tuple([len(src)]+list(im.shape)))
    for i in range(len(src)):
        memmap[i,...] = tifffile.imread(src[i])
        memmap.flush()

    return dst

def view_brain(vol, subsections=5, save=False, dpi=500, cmap = 'gray'):
    '''Function to visualize the brain

    Inputs
    ------------
    vol = volume (pth or array)
    subsections = number of subsections to divide the brain into
    save = False: show the image
           str: save location
    dpi = dots per inch to save/view
    '''
    from skimage.external import tifffile
    import matplotlib.pyplot as plt
    if type(vol) == str: vol = tifffile.imread(vol)

    vol = norm(vol)
    plt.figure(figsize=(15,5))
    for i in range(subsections):
        step = vol.shape[0] / subsections
        ax = plt.subplot(1,subsections,i+1)
        plt.imshow(np.max(vol[i*step:(i+1)*step], axis=0), cmap=cmap)
        ax.axis('off');

    if save:
        plt.savefig(save, dpi=dpi, transparent=True)
    else:
        plt.show()

    return


def norm(im, percentile=(0.1, 99.9), outdepth='uint16', zero_extremes=False, verbose=False):
    '''Function to clip percentiles and then normalize the image

    im = np.array
    percentile: tuple of lower and upper clipping percentiles
                False: no clipping
    outdepth: bitdepth to output image as
    zero_extremes False: values outside upper and lower percentiles become upper and lower
                  True: values outside upper and lower percentiles become 0
    '''
    from tools.utils.io import change_bitdepth
    im = np.copy(im.astype('float'))
    if percentile:
        low, high = np.percentile(im, percentile)
    else:
        low = np.min(im)
        high = np.max(im)

    if zero_extremes:
        low_out = 0
        high_out = 0
    else:
        low_out = low
        high_out = high

    im[im<low] = low_out
    im[im>high] = high_out
    im = (im - low) / (high - low)
    if verbose and percentile: print('{} pixel intensity at {} percentile, {} pixel intensity at {} percentile'.format(low, percentile[0], high, percentile[1]))

    return change_bitdepth(im, outdepth=outdepth)

def convert_to_mhd(src, dims, dst=False, verbose = False):
    '''Function to change image from np array(tif) into MHD with appropriate spacing.

    Inputs
    ------------------
    src = pth to tiff file
    dims = tuple of um/pixel resolution in *******XYZ********; i.e. (25,25,25)
    dst = (optional) file name and path if requiring different than src+'.mhd'

    Returns
    ------------------
    pth to saved mhd file
    '''

    #im = sitk.GetImageFromArray(tifffile.imread(src))
    import SimpleITK as sitk
    im = sitk.ReadImage(src)

    #print im
    im.SetSpacing(dims)
    if src[-4:] == '.tif': src=src[:-4]

    if dst:
        if dst[-4:] != '.mhd': dst=dst+'.mhd'
        sitk.WriteImage(im, dst)
        if verbose: print(dst)
        return dst
    else:
        sitk.WriteImage(im, src+'.mhd')
        if verbose: print(src+'.mhd')
        return src+'.mhd'

import filecmp
import os.path

def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
   """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only)>0 or len(dirs_cmp.right_only)>0 or \
        len(dirs_cmp.funny_files)>0:
        return False
    (_, mismatch, errors) =  filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch)>0 or len(errors)>0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def log_step(task, out='complete', **kwargs):
    '''Function to log completion of a step

    Assumes a dictionary as input.
    If dictionary contains k,v: 'dict_path'=path_to_save_location, then will automatically save
    '''
    if 'tasks' not in kwargs: kwargs['tasks']={}
    kwargs['tasks'][task]=out

    #save out...
    if 'dict_path' in kwargs: save_kwargs(kwargs['dict_path'], **kwargs)

    return kwargs
def check_step(task, **kwargs):
    '''Function to log completion of a step if step doesn't already == 'complete'

    Assumes a dictionary as input.
    If dictionary contains k,v: 'dict_path'=path_to_save_location, then will automatically save
    '''
    if 'tasks' not in kwargs: kwargs['tasks']={}
    if task in kwargs['tasks'] and kwargs['tasks'][task] == 'complete':
        return True
    else:
        return False
    
from subprocess import check_output

def sp_call(call):
    print(check_output(call, shell=True))
    return
