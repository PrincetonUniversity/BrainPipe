#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:54:29 2019

@author: wanglab
"""

import os
import numpy as np
from skimage.external import tifffile
import matplotlib as mpl
from tools.imageprocessing.orientation import fix_orientation
from tools.registration.transform import count_structure_lister, transformed_pnts_to_allen_helper_func
from tools.utils.io import load_kwargs, listdirfull, makedir
from collections import Counter
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()


def orientation_crop_check(src, axes=('0', '1', '2'), crop=False, dst=False):
    '''Function to check orientation and cropping. MaxIPs along 0 axis.

      'crop': #use to crop volume, values below assume horizontal imaging and sagittal atlas
                False
                cerebellum: '[:,390:,:]'
                caudal midbrain: '[:,300:415,:]'
                midbrain: '[:,215:415,:]'
                thalamus: '[:,215:345,:]'
                anterior cortex: '[:,:250,:]'
    'dst': (optional) path+extension to save image

    Returns
    ---------------
    cropped image

    '''
    fig = plt.figure()
    plt.axis('off')
    ax = fig.add_subplot(1, 2, 1)
    if type(src) == str:
        src = tifffile.imread(src)
    plt.imshow(np.max(src, axis=0))
    plt.title('Before reorientation')

    ax = fig.add_subplot(1, 2, 2)
    if crop:
        src = eval('src{}'.format(crop))
    src = fix_orientation(src, axes=axes)
    plt.imshow(np.max(src, axis=0))
    plt.title('After reorientation')

    if dst:
        plt.savefig(dst, dpi=300)
    return src


def optimize_inj_detect(src, threshold=3, filter_kernel=(3, 3, 3), dst=False):
    '''Function to test detection parameters

    'dst': (optional) path+extension to save image

    '''
    if type(src) == str:
        src = tifffile.imread(src)
    arr = find_site(src, thresh=threshold, filter_kernel=filter_kernel)*45000
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(np.max(arr, axis=0))
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(np.max(src, axis=0), cmap='jet')
    plt.axis('off')

    if dst:
        plt.savefig(dst, dpi=300)

    return


def pool_injections_for_analysis(**kwargs):

    inputlist = kwargs['inputlist']
    dst = kwargs['dst']
    makedir(dst)
    injscale = kwargs['injectionscale'] if 'injectionscale' in kwargs else 1
    imagescale = kwargs['imagescale'] if 'imagescale' in kwargs else 1
    axes = kwargs['reorientation'] if 'reorientation' in kwargs else ('0', '1', '2')
    cmap = kwargs['colormap'] if 'colormap' in kwargs else 'plasma'
    id_table = kwargs['id_table'] if 'id_table' in kwargs else '/jukebox/temp_wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx'
    save_tif = kwargs['save_tif'] if 'save_tif' in kwargs else False
    num_sites_to_keep = kwargs['num_sites_to_keep'] if 'num_sites_to_keep' in kwargs else 1
    nonzeros = []
    ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotation']))
    if kwargs['crop']:
        ann = eval('ann{}'.format(kwargs['crop']))
    allen_id_table = pd.read_excel(id_table)

    for i in range(len(inputlist)):
        impth = inputlist[i]
        animal = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(impth))))

        print('\n\n_______\n{}'.format(animal))

        print('  loading:\n     {}'.format(animal))
        im = tifffile.imread(impth)

        if kwargs['crop']:
            im = eval('im{}'.format(kwargs['crop']))  # ; print im.shape

        # reorient to coronal?

        # segment
        arr = find_site(im, thresh=kwargs['threshold'], filter_kernel=kwargs['filter_kernel'],
                        num_sites_to_keep=num_sites_to_keep)*injscale
        if save_tif:
            tifffile.imsave(os.path.join(dst, '{}'.format(
                os.path.dirname(impth))+'_inj.tif'), arr.astype('float32'))

        # optional 'save_individual'
        if kwargs['save_individual']:
            im = im*imagescale
            a = np.concatenate((np.max(im, axis=0), np.max(arr.astype('uint16'), axis=0)), axis=1)
            b = np.concatenate((np.fliplr(np.rot90(np.max(fix_orientation(im, axes=axes), axis=0), k=3)), np.fliplr(
                np.rot90(np.max(fix_orientation(arr.astype('uint16'), axes=axes), axis=0), k=3))), axis=1)
            plt.figure()
            plt.imshow(np.concatenate((b, a), axis=0), cmap=cmap, alpha=1)
            plt.axis('off')
            plt.savefig(os.path.join(dst, '{}'.format(animal)+'.pdf'), dpi=300, transparent=True)
            plt.close()

        # cell counts to csv
        print('   finding nonzero pixels for voxel counts...')
        nz = np.nonzero(arr)
        nonzeros.append(zip(*nz))  # <-for pooled image
        pos = transformed_pnts_to_allen_helper_func(np.asarray(zip(*[nz[2], nz[1], nz[0]])), ann)
        tdf = count_structure_lister(allen_id_table, *pos)
        if i == 0:
            df = tdf.copy()
            countcol = 'count' if 'count' in df.columns else 'cell_count'
            df.drop([countcol], axis=1, inplace=True)
        df[animal] = tdf[countcol]

    df.to_csv(os.path.join(dst, 'voxel_counts.csv'))
    print('\n\nCSV file of cell counts, saved as {}\n\n\n'.format(
        os.path.join(dst, 'voxel_counts.csv')))

    # condense nonzero pixels
    # this list has duplicates if two brains had the same voxel w label
    nzs = [str(x) for xx in nonzeros for x in xx]
    c = Counter(nzs)
    array = np.zeros(im.shape)
    print('Collecting nonzero pixels for pooled image...')
    tick = 0
    # generating pooled array where voxel value = total number of brains with that voxel as positive
    for k, v in c.iteritems():
        k = [int(xx) for xx in k.replace('(', '').replace(')', '').split(',')]
        array[k[0], k[1], k[2]] = int(v)
        tick += 1
        if tick % 50000 == 0:
            print('   {}'.format(tick))

    # load atlas and generate final figure
    print('Generating final figure...')
    atlas = tifffile.imread(kwargs['atlas'])
    arr = fix_orientation(array, axes=axes)
    # cropping
    # if 'crop_atlas' not in kwargs:
    if kwargs['crop']:
        atlas = eval('atlas{}'.format(kwargs['crop']))
    atlas = fix_orientation(atlas, axes=axes)

    my_cmap = eval('plt.cm.{}(np.arange(plt.cm.RdBu.N))'.format(cmap))
    my_cmap[:1, :4] = 0.0
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    my_cmap.set_under('w')
    plt.figure()
    plt.imshow(np.max(atlas, axis=0), cmap='gray')
    plt.imshow(np.max(arr, axis=0), alpha=0.99, cmap=my_cmap)
    plt.colorbar()
    plt.axis('off')
    dpi = int(kwargs['dpi']) if 'dpi' in kwargs else 300
    plt.savefig(os.path.join(dst, 'heatmap.pdf'), dpi=dpi, transparent=True)
    plt.close()

    print('Saved as {}'.format(os.path.join(dst, 'heatmap.pdf')))

    return df


def find_site(im, thresh=10, filter_kernel=(5, 5, 5), num_sites_to_keep=1):
    """Find a connected area of high intensity, using a basic filter + threshold + connected components approach

    by: bdeverett

    Parameters
    ----------
    img : np.ndarray
        3D stack in which to find site (technically need not be 3D, so long as filter parameter is adjusted accordingly)
    thresh: float
        threshold for site-of-interest intensity, in number of standard deviations above the mean
    filter_kernel: tuple
        kernel for filtering of image before thresholding
    num_sites_to_keep: int, number of injection sites to keep, useful if multiple distinct sites

    Returns
    --------
    bool array of volume where coordinates where detected
    """
    from scipy.ndimage.filters import gaussian_filter as gfilt
    from scipy.ndimage import label
    if type(im) == str:
        im = tifffile.imread(im)

    filtered = gfilt(im, filter_kernel)
    thresholded = filtered > filtered.mean() + thresh*filtered.std()
    labelled, nlab = label(thresholded)

    if nlab == 0:
        raise Exception('Site not detected, try a lower threshold?')
    elif nlab == 1:
        return labelled.astype(bool)
    elif num_sites_to_keep == 1:
        sizes = [np.sum(labelled == i) for i in range(1, nlab+1)]
        return labelled == np.argmax(sizes)+1
    else:
        sizes = [np.sum(labelled == i) for i in range(1, nlab+1)]
        vals = [i+1 for i in np.argsort(sizes)[-num_sites_to_keep:][::-1]]
        return np.in1d(labelled, vals).reshape(labelled.shape)


# %%
if __name__ == '__main__':

    # check if reorientation is necessary
    #    src = "/home/wanglab/mounts/wang/Jess/lightsheet_output/201904_ymaze_cfos/injection/processed/an23/elastix/an23_ymazefos_021519_1d3x_647_008na_1hfds_z10um_400msec_ch00/result.1.tif"
    #    src = orientation_crop_check(src, axes = ('2','1','0'), crop = '[:,450:,:]')
    #
    #    #optimize detection parameters for inj det
    #    optimize_inj_detect(src, threshold=3, filter_kernel = (5,5,5))

    # run
    # suggestion: save_individual=True,
    # then inspect individual brains, which you can then remove bad brains from list and rerun function
    brains = ['an26']
#             'an28',
#             'an22',
#             'an23',
#             'an25',
#             'an27',
#             'an19',
#             'an20',
#             'an21',
#             'an24']

    pth = "/home/wanglab/mounts/wang/Jess/lightsheet_output/201904_ymaze_cfos/injection/processed"
    brains = [os.path.join(pth, xx) for xx in brains]
    inputlist = []
    for brain in brains:
        reg_pth = os.path.join(pth, os.path.join(brain, "elastix"))
        inj_pth = [os.path.join(reg_pth, xx) for xx in os.listdir(
            reg_pth) if os.path.isdir(os.path.join(reg_pth, xx))][0]
        inputlist.append(os.path.join(inj_pth, "result.1.tif"))

    kwargs = {'inputlist': inputlist,
              'filter_kernel': (5, 5, 5),
              'threshold': 3,
              'num_sites_to_keep': 1,
              'injectionscale': 45000,
              'imagescale': 3,
              'reorientation': ('2', '0', '1'),
              'crop': '[:,500:,:200]',
              'dst': '/home/wanglab/Desktop/test',
              'save_individual': True,
              'colormap': 'plasma',
              'atlas': "/home/wanglab/mounts/LightSheetTransfer/atlas/sagittal_atlas_20um_iso.tif",
              'annotation': "/home/wanglab/mounts/LightSheetTransfer/atlas/annotation_sagittal_atlas_20um_iso.tif",
              'id_table': "/home/wanglab/mounts/LightSheetTransfer/atlas/ls_id_table_w_voxelcounts.xlsx"
              }

    df = pool_injections_for_analysis(**kwargs)
