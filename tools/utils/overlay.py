#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:11:08 2017

@author: tpisano
"""

from tools.utils.io import listdirfull, makedir
import matplotlib as mpl
from skimage.external import tifffile
import SimpleITK as sitk
from tools.imageprocessing.orientation import fix_orientation
import numpy as np, matplotlib.pyplot as plt, os
from math import ceil

def tile(src, subtitles=False, subplots=False, cmap=plt.cm.jet, method='matplotlib', dst = False, dpi=300, fontsize=12, figsize=False, colorbar=False):
    '''Function to handle tiling
    
    Inputs
    --------------
    src = list of np arrays
    subplots(optional): shape to use for subplots
    cmap = colormap
    method = matplotlib tiling using suplots
            else: concatenate image based on subplots or evenly (not functional)
    titles (optional) = subtitles to add
    dst (optional) save
    fontsize=12
    figsize=False
    colorbar=T/F
    '''
    
    if not subplots: subplots = [int(ceil(len(src) **.5))]*2
    if not subtitles: subtitles = ['' for xx in range(len(src))]
    
    if method == 'matplotlib':
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        for i in range(len(src)):
            ax = fig.add_subplot(subplots[0], subplots[1], i+1)
            if not colorbar: plt.imshow(src[i], cmap=cmap); plt.axis('off')
            if colorbar: plt.imshow(src[i], cmap=cmap); plt.colorbar()
            ax.set_title(subtitles[i], fontdict={'fontsize': fontsize})
        if dst:
            plt.savefig(dst, dpi=dpi, transparent=True)
            plt.close()
            print ('saved as {}'.format(dst))
            return
        return fig
        
    else: #not functional
        tick=0
        dim = int(ceil(len(animals)**.5))
        size = 2*dim
        fig = plt.figure(figsize=(size,size))
        biga = np.zeros((dim*y, dim*x))
        im = 1
        im[tick/dim*y : (tick+dim)/dim*y, (tick%dim)*x : ((tick%dim)+1)*x] = a
        return
    
    
    '''
        ndst = dst+'/volumetric_reconstruction'; makedir(ndst)
    tick=0
    xtile=4
    ytile=4
    for i in range(0, len(gt), xtile*ytile):
        scale_factor=1.5
        lst=[]
        for xy in range(ytile*xtile):
            ii=xy+i
            lst.append(np.concatenate([norm(adjust_boundary(inn[ii,:,:,:,0], gt[ii,:,:,:,0].shape))*scale_factor, norm(gt[ii,:,:,:,0]), pred[ii,:,:,:,0]*scale_factor], axis=2).astype('float32'))
        nlst=[]
        for y in range(ytile):
            print y*xtile, (y+1)*xtile
            print
            nlst.append(np.concatenate(lst[y*xtile:(y+1)*xtile], axis=1))
        tifffile.imsave(os.path.join(ndst, '{}-{}.tif'.format(tick,tick+(xtile*ytile))), np.concatenate(nlst,axis=2))
        tick+=1
        if tick%10==0: print('{} of {}'.format(tick, inn.shape[0]))
    
    
    '''
    
def cleared_brain_qc_control(src, dst, nm, cmap='viridis'):
    '''src = volume to process
    '''
    
    im = tifffile.imread(src)
    up = np.percentile(im, 99.9)
    down = np.percentile(im, 0.1)
    im[im < down] = down
    im[im>up] = up
    high = np.max(im)
    low = np.min(im)
    
    im = (im - low) / (high - low)
  
    im = np.swapaxes(im, 0, 2)
    plt.figure(figsize=(15,5))
    for i in range(5):
        step = im.shape[0] / 5
        ax = plt.subplot(1,5,i+1)
        plt.imshow(np.max(im[i*step:(i+1)*step], axis=0), cmap=cmap)
        ax.axis('off');
    plt.title(nm);
    plt.savefig(os.path.join(dst, nm+'.png'), dpi=300)
    return


def overlay(fl, out, atl = '/home/wanglab/wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif', axes = ('2','0','1'), crop='-110:,:,:', cmap='viridis', dpi=500):
    '''Function to generate coronal and horizontal overlays
    
    Inputs:
        -----------
    fl: path to tiff volume
    out: path, loc, ext to save
    atl (optional) atlas tif file
    axes (optional for rotation of atl): ('2','0','1')
    crop(optional) dims to crop atlas e.g. '-110:,:,:'
    cmap (optional) colormap, e.g. 'viridis'
    dpi (optional) e.g. 500
    '''
    atl = fix_orientation(tifffile.imread(atl), axes=axes)
    atl = eval('atl[{}]'.format(crop))
    array = tifffile.imread(fl)
    my_cmap = eval('plt.cm.{}(np.arange(plt.cm.RdBu.N))'.format(cmap))
    my_cmap[:1,:4] = 0.0  
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    my_cmap.set_under('w')
    plt.figure()
    #plt.imshow(np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0))
    #plt.figure()
    #plt.imshow(np.max(atl, axis=0))
    a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
    #MEAN*****
    b=np.concatenate((np.mean(fix_orientation(array, axes=('0','2', '1')), axis=0), np.mean(array, axis=0)), axis=0)
    plt.imshow(a, cmap='gray')
    plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
    plt.colorbar()
    plt.savefig('{}.pdf'.format(out), dpi=dpi, transparent=True)
    plt.close()    
    
    return


#%%good starts here I think:
if __name__ == "__main__": #for aleksandra's paper
    
    #rename
    fls = [os.rename(xx, xx[:-1]) for xx in listdirfull('/home/wanglab/wang/Talmo/data/DREADDs/TV/tp') if '.tiff' in xx]
    #start    
    atl = fix_orientation(tifffile.imread('/home/wanglab/wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif'), axes=('2','0','1'))
    atl = atl[-110:]
    fls = [xx for xx in listdirfull('/home/wanglab/wang/Talmo/data/DREADDs/TV/tp') if '.tif' in xx]
    out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/orig_8bit_mean_per_brain_individuals'
    os.mkdir(out)
    
    for fl in fls:
        print fl
        array = tifffile.imread(fl)#.astype('bool')
        my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
        my_cmap[:1,:4] = 0.0  
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        my_cmap.set_under('w')
        plt.figure()
        a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
        #MEAN*****
        b=np.concatenate((np.mean(fix_orientation(array, axes=('0','2', '1')), axis=0), np.mean(array, axis=0)), axis=0)
        plt.imshow(a, cmap='gray')
        plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
        plt.colorbar()
        plt.savefig(out+'/{}.pdf'.format(os.path.basename(fl)), dpi=500, transparent=True)
        plt.close()
    out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/mean_per_brain_individuals'
    os.mkdir(out)
    
    for fl in fls:
        print fl
        array = tifffile.imread(fl).astype('bool')
        my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
        my_cmap[:1,:4] = 0.0  
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        my_cmap.set_under('w')
        plt.figure()
        a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
        #MEAN*****
        b=np.concatenate((np.mean(fix_orientation(array, axes=('0','2', '1')), axis=0), np.mean(array, axis=0)), axis=0)
        plt.imshow(a, cmap='gray')
        plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
        plt.colorbar()
        plt.savefig(out+'/{}.pdf'.format(os.path.basename(fl)), dpi=500, transparent=True)
        plt.close()
        
    out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/sum_per_brain_individuals'
    os.mkdir(out)
    
    for fl in fls:
        print fl
        array = tifffile.imread(fl).astype('bool')
        my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
        my_cmap[:1,:4] = 0.0  
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        my_cmap.set_under('w')
        plt.figure()
        a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
        #MEAN*****
        b=np.concatenate((np.sum(fix_orientation(array, axes=('0','2', '1')), axis=0), np.sum(array, axis=0)), axis=0)
        plt.imshow(a, cmap='gray')
        plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
        plt.colorbar()
        plt.savefig(out+'/{}.pdf'.format(os.path.basename(fl)), dpi=500, transparent=True)
        plt.close()
#%%
    #GROUPS:
    aa=[]; bb=[]; ageloc = []
    from tools.utils.io import listdirfull, makedir
    fls = [xx for xx in listdirfull('/home/wanglab/wang/Talmo/data/DREADDs/TV/tp') if '.tif' in xx]
    df = pd.read_excel('/home/wanglab/wang/Talmo/data/DREADDs/TV/List_allmice.xlsx')
    df = df.dropna()
    for age in df['Age'].unique():
        for loc in df['Injection'].unique():
            print age, loc
            animals = df[(df.Age==age) & (df.Injection == loc) & (df.DREADD == 'Inhibitory ')]
            if len(animals)>0:
            
                #COMBINE ---mean mean
                #find files
                #rotate to correct orientation (coronal or horziotonal) then sum each brain and then average them all
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/voxel-by-voxelsum_maxip'; makedir(out)
                array = np.asarray([tifffile.imread(fl).astype('bool') for an in animals.Animal.tolist() for fl in fls if str(an) in fl])
                my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                my_cmap[:1,:4] = 0.0  
                my_cmap = mpl.colors.ListedColormap(my_cmap)
                my_cmap.set_under('w')
                plt.figure()
                a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                #MEAN_mean
                hor_array = np.asarray([fix_orientation(im, axes=('0','2', '1')) for im in array])
                b=np.concatenate((np.max(np.sum(hor_array, axis=0), axis=0), np.max(np.sum(array, axis=0), axis=0)), axis=0)
                aa.append(a)
                bb.append(b)
                ageloc.append(age+'_'+loc)
                plt.imshow(a, cmap='gray')
                plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                plt.savefig(out+'/{} {}.pdf'.format(loc, age), dpi=500, transparent=True)
                plt.close()
            

    aaa = np.concatenate(aa, axis=0)
    bbb = np.concatenate(bb, axis=0)
    plt.imshow(aaa, cmap='gray')
    plt.imshow(bbb, alpha=.9, cmap=my_cmap);  plt.axis('off')
    plt.colorbar()
    plt.savefig(out+'/ALL_top_to_bottom_order_{}.pdf'.format(''.join(ageloc)), dpi=1500, transparent=True)
    plt.close()            
            
            
            #%%
                #GROUPS:
    from tools.utils.io import listdirfull
    fls = [xx for xx in listdirfull('/home/wanglab/wang/Talmo/data/DREADDs/TV/tp') if '.tif' in xx]
    df = pd.read_excel('/home/wanglab/wang/Talmo/data/DREADDs/TV/List_allmice.xlsx')
    df = df.dropna()
    for age in df['Age'].unique():
        for loc in df['Injection'].unique():
            print age, loc
            animals = df[(df.Age==age) & (df.Injection == loc) & (df.DREADD == 'Inhibitory ')]
            if len(animals)>0:
                #tiled_sum
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/tiling_sum_per_brain'; makedir(out)
                tick=0
                y = 430
                x = 456
                dim = int(ceil(len(animals)**.5))
                size = 2*dim
                fig = plt.figure(figsize=(size,size))
                biga = np.zeros((dim*y, dim*x))
                bigb = np.zeros((dim*y, dim*x))
                names = []
                for idx, row in animals.iterrows():
                    array = [tifffile.imread(fl).astype('bool') for fl in fls if str(row['Animal']) in fl][0]
                    names.append(str(row['Animal']))
                    my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                    my_cmap[:1,:4] = 0.0  
                    my_cmap = mpl.colors.ListedColormap(my_cmap)
                    my_cmap.set_under('w')
                    a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                    b=np.concatenate((np.sum(fix_orientation(array, axes=('0','2', '1')), axis=0), np.sum(array, axis=0)), axis=0)
                    biga[tick/dim*y : (tick+dim)/dim*y, (tick%dim)*x : ((tick%dim)+1)*x] = a
                    bigb[tick/dim*y : (tick+dim)/dim*y, (tick%dim)*x : ((tick%dim)+1)*x] = b
                    tick+=1
                plt.imshow(biga, cmap='gray')
                plt.imshow(bigb, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                for tick in range(len(names)):
                    yshift=20
                    plt.text((tick%dim)*x,(tick+dim)/dim*y-yshift,'{}'.format(names[tick]), size=10, color='w')
                plt.savefig(out+'/{} {}_tiled.pdf'.format(loc, age), dpi=1000, transparent=True)
                plt.close()     
                #tiled_mean ###IMPORTANT TO MAKE SINGLE IMAGE THEN IMSHOW TO FORCE COLORMAP TO BE THE SAME FOR ALL BRAINS
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/tiling_mean_per_brain'; makedir(out)
                tick=0
                y = 430
                x = 456
                dim = int(ceil(len(animals)**.5))
                size = 2*dim
                fig = plt.figure(figsize=(size,size))
                biga = np.zeros((dim*y, dim*x))
                bigb = np.zeros((dim*y, dim*x))
                names = []
                for idx, row in animals.iterrows():
                    array = [tifffile.imread(fl).astype('bool') for fl in fls if str(row['Animal']) in fl][0]
                    names.append(str(row['Animal']))
                    my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                    my_cmap[:1,:4] = 0.0  
                    my_cmap = mpl.colors.ListedColormap(my_cmap)
                    my_cmap.set_under('w')
                    a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                    b=np.concatenate((np.mean(fix_orientation(array, axes=('0','2', '1')), axis=0), np.mean(array, axis=0)), axis=0)
                    biga[tick/dim*y : (tick+dim)/dim*y, (tick%dim)*x : ((tick%dim)+1)*x] = a
                    bigb[tick/dim*y : (tick+dim)/dim*y, (tick%dim)*x : ((tick%dim)+1)*x] = b
                    tick+=1
                plt.imshow(biga, cmap='gray')
                plt.imshow(bigb, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                for tick in range(len(names)):
                    yshift=20
                    plt.text((tick%dim)*x,(tick+dim)/dim*y-yshift,'{}'.format(names[tick]), size=10, color='w')
                plt.savefig(out+'/{} {}_tiled.pdf'.format(loc, age), dpi=1000, transparent=True)
                plt.close()                    
                #COMBINE
                #OLD
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/old_sum_mean_8bit'; makedir(out)
                array = np.sum(np.asarray([tifffile.imread(fl) for an in animals.Animal.tolist() for fl in fls if str(an) in fl]), axis=0)
                my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                my_cmap[:1,:4] = 0.0  
                my_cmap = mpl.colors.ListedColormap(my_cmap)
                my_cmap.set_under('w')
                plt.figure()
                a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                #MEAN*****
                b=np.concatenate((np.mean(fix_orientation(array, axes=('0','2', '1')), axis=0), np.mean(array, axis=0)), axis=0)
                plt.imshow(a, cmap='gray')
                plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                plt.savefig(out+'/{} {}.pdf'.format(loc, age), dpi=500, transparent=True)

                #COMBINE ---mean mean
                #find files
                #rotate to correct orientation (coronal or horziotonal) then sum each brain and then average them all
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/mean_each_brain_mean_across_brain'; makedir(out)
                array = np.asarray([tifffile.imread(fl).astype('bool') for an in animals.Animal.tolist() for fl in fls if str(an) in fl])
                my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                my_cmap[:1,:4] = 0.0  
                my_cmap = mpl.colors.ListedColormap(my_cmap)
                my_cmap.set_under('w')
                plt.figure()
                a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                #MEAN_mean
                hor_array = np.asarray([fix_orientation(im, axes=('0','2', '1')) for im in array])
                b=np.concatenate((np.mean(np.mean(hor_array, axis=1), axis=0), np.mean(np.mean(array, axis=1), axis=0)), axis=0)
                plt.imshow(a, cmap='gray')
                plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                plt.savefig(out+'/{} {}.pdf'.format(loc, age), dpi=500, transparent=True)
                plt.close()

                #COMBINE ---mean mean
                #find files
                #rotate to correct orientation (coronal or horziotonal) then sum each brain and then average them all
                out = '/home/wanglab/wang/Talmo/data/DREADDs/TV/overlays/sum_each_brain_mean_across_brain'; makedir(out)
                array = np.asarray([tifffile.imread(fl).astype('bool') for an in animals.Animal.tolist() for fl in fls if str(an) in fl])
                my_cmap = plt.cm.jet(np.arange(plt.cm.RdBu.N))
                my_cmap[:1,:4] = 0.0  
                my_cmap = mpl.colors.ListedColormap(my_cmap)
                my_cmap.set_under('w')
                plt.figure()
                a=np.concatenate((np.max(fix_orientation(atl, axes=('0','2', '1')), axis=0), np.max(atl, axis=0)), axis=0)
                #SUM_MEAN
                hor_array = np.asarray([fix_orientation(im, axes=('0','2', '1')) for im in array])
                b=np.concatenate((np.mean(np.sum(hor_array, axis=1), axis=0), np.mean(np.sum(array, axis=1), axis=0)), axis=0)
                plt.imshow(a, cmap='gray')
                plt.imshow(b, alpha=.9, cmap=my_cmap);  plt.axis('off')
                plt.colorbar()
                plt.savefig(out+'/{} {}.pdf'.format(loc, age), dpi=500, transparent=True)
                plt.close()                
