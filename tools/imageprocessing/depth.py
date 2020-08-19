import os
import matplotlib.pyplot as pl
pl.ioff() 
import numpy as np
import tifffile
from matplotlib import gridspec
from scipy.ndimage.interpolation import zoom


def layout(path0,path1,path2,dest_path):
    pl.ioff() #for spock    
    im0,im1,im2 = pl.imread(path0),pl.imread(path1),pl.imread(path2)
    gs = gridspec.GridSpec(2, 2, width_ratios=[im2.shape[1], im0.shape[1]], height_ratios=[im0.shape[0], im1.shape[0]], hspace=0.02, wspace=0.02)

    for axi,im in zip([0,1,-1],[im2,im0,im1]):
        ax = pl.subplot(gs[axi])
        ax.imshow(im)
        ax.axis('off')
    pl.savefig(os.path.join(dest_path, 'summary.png'))
    pl.close()
    return #TP added to help with spock issues

def overlay(bg_path, mask_path, dest_path, thresh=0.05, alpha=1):
    pl.ioff() #for spock
    # load bg
    bg = pl.imread(bg_path)
    bg = bg[...,:-1].sum(axis=-1)
    bg = np.repeat(bg[:,:,None], 4, axis=-1)
    bg = (bg-bg.min())/(bg.max()-bg.min())
    bg[:,:,-1] = 1.0
    
    # load mask
    mask = pl.imread(mask_path)
    mask[...,:-1] = (mask[...,:-1] - mask[...,:-1].min())/(mask[...,:-1].max()-mask[...,:-1].min())
    mask = zoom(mask, np.asarray(bg.shape)/np.asarray(mask.shape))

    result = bg
    idxs = np.where(mask[:,:,:-1].sum(axis=-1)>thresh)

    toadd = mask[idxs[0],idxs[1],:]
    toadd[...,-1] = alpha
    result[idxs[0],idxs[1]] *= toadd
    pl.imsave(dest_path, result)
    #return mask,result #TP REMOVING SINCE IT SEEMS TO MESS UP SPOCK
    return

def colorcode(src_path, dest_path, cmap=pl.cm.jet, show_slices=False):
    """Color a 3D stack by depth

    Parameters
    ----------
    src_path : str
        path to source tiff file
    dest_path : str
        path for saving output (a directory)
    cmap : matplotlib colormap
        for colouring
    show_slices : bool
        show a figure with individually coloured slices 
    """
    pl.ioff()    
    im = tifffile.imread(src_path)
    im = np.swapaxes(im, 2, 0)
    im = im.astype(float)
    im = (im-im.min())/(im.max()-im.min()) #normalize
    im_new = im[...,None] * cmap(np.linspace(0,1,len(im)))[:,None,None] #multiply depth by color 
    im_new[:,:,:,-1] = 1 #reset alpha values to always be 1
    
    #max0 = im_new.max(axis=0)
    #max1 = im_new.max(axis=1)
    #max2 = im_new.max(axis=2)
    #max2 = ndimage.rotate(max2, -90)
    #max1 = np.repeat(max1,3,axis=0)
    #max2 = np.repeat(max2,3,axis=1)
    
    ##TP CHANGED; ORIG IS ABOVE    
    max0 = im_new.max(axis=0)
    max1 = im_new.max(axis=1) 
    max2 = im_new.max(axis=2) 
    max1 = np.flipud(max1)
    
    max2 = np.fliplr(np.swapaxes(max2, 1, 0))

    
    gs = gridspec.GridSpec(2, 2, width_ratios=[max2.shape[1], max0.shape[1]], height_ratios=[max0.shape[0], max1.shape[0]], hspace=0.02, wspace=0.01)

    for axi,im in zip([0,1,-1],[max2,max0,max1]): ###changed...it was 0,1,2 I think
        ax = pl.subplot(gs[axi])
        ax.imshow(im)
        ax.axis('off')
    pl.savefig(os.path.join(dest_path, 'composite.png'))
    pl.close()
    pl.imsave(os.path.join(dest_path, 'proj0.png'), max0)
    pl.imsave(os.path.join(dest_path, 'proj1.png'), max1)
    pl.imsave(os.path.join(dest_path, 'proj2.png'), max2)
    #return im_new #TP REMOVING SINCE IT SEEMS TO MESS UP SPOCK
    return
        
    if show_slices:
        n = int(np.ceil(np.sqrt(len(im_new))))
        fig,axs = pl.subplots(n,n,squeeze=True,num='Sections')
        for ax,i in zip(axs.flat,im_new):
            ax.imshow(i)
        [ax.axis('off') for ax in axs.flat]
