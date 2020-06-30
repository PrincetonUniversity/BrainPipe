#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:21:22 2019

@author: wanglab
"""

import matplotlib.pyplot as plt, os, matplotlib as mpl, numpy as np
from skimage.external import tifffile

tab20cmap = [plt.cm.tab20(xx) for xx in range(20)]
tab20cmap_nogray = tab20cmap[:14] + tab20cmap[16:]
tab20cmap_nogray = mpl.colors.ListedColormap(tab20cmap_nogray, 
                                             name = "tab20cmap_nogray")

def make_2D_composite_overlay_of_heatmaps(src, atl_pth, ann_pth, allen_id_table, 
                                save_dst, condition,
                                positive = True, negative = True,
                                zstep = 20, colorbar_cutoff = 65):
    
    """ custom script for cfos visualisation """
    print(save_dst)

    #half brain        
    #read variables
    vol = tifffile.imread(src)
    atl = tifffile.imread(atl_pth)
    ann = tifffile.imread(ann_pth)
    
    pvol = vol[:,:,:,1]
    nvol = vol[:,:,:,0]
    atl = np.rot90(np.transpose(atl, [1, 0, 2]), axes = (2,1)) #sagittal to coronal
    ann = np.rot90(np.transpose(ann, [1, 0, 2]), axes = (2,1)) #sagittal to coronal
    
    #cut everything in half the same way
    pvol = pvol[:, :, int(pvol.shape[2]/2):]
    nvol = nvol[:, :, int(nvol.shape[2]/2):]
    atl = atl[:, :, int(atl.shape[2]/2):]
    ann = ann[:, :, int(ann.shape[2]/2):]
    
    assert atl.shape == ann.shape == nvol.shape
    
    #threshold values
    pvol[pvol!=0.0] = 1.0
    nvol[nvol!=0.0] = 1.0
    
    rngs = range(0, atl.shape[0], zstep)
    
    #positive
    if positive:
        print("\npositive")
        #make fig
        #pvals
        plt.style.use("dark_background")
        fig, axs = plt.subplots(4, 7, sharex="col", sharey="row",
                    gridspec_kw={"hspace": 0, "wspace": 0}, facecolor="black",
                    figsize = (6, 4))                    
        #modify colormap
        my_cmap = plt.cm.viridis(np.arange(plt.cm.RdBu.N))
        my_cmap[:colorbar_cutoff,:4] = 0.0
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        my_cmap.set_under("w")       
        #plot
        i = 0
        for row in axs:
            for col in row:
                try:
                    img = col.imshow(np.max(atl[rngs[i]:rngs[i+1]], axis=0), cmap="gray", alpha=1)
                    img = col.imshow(np.sum(pvol[rngs[i]:rngs[i+1]], axis=0), cmap=my_cmap, alpha=0.95, vmin=0, vmax=zstep)
                    col.axis("off")
                    plt.tight_layout()
                except:
                     col.imshow(np.zeros_like(np.max(atl[rngs[0]:rngs[1]], axis=0)), cmap = "binary_r")
                     col.axis("off")
                     plt.tight_layout()    
                i += 1        
        #custom colorbar
        cbar = fig.colorbar(img, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01, ticks = [5, 10, 15, 20])
        cbar.set_label("# of planes represented",size=7)
        # access to cbar tick labels:
        cbar.ax.tick_params(labelsize=5) 
        cbar.ax.set_yticklabels(["5", "10", "15", "20"])  # vertically oriented colorbar
        
        plt.savefig(os.path.join(save_dst, "{}_pos_correlated_voxels_half_brain.pdf".format(condition)), bbox_inches = "tight", dpi=300, 
                    facecolor=fig.get_facecolor(), edgecolor="none", transparent = True)
        plt.close()

    #negative
    if negative:        
        print("\nnegative")        
        #make fig
        #pvals
        plt.style.use("dark_background")
        fig, axs = plt.subplots(4, 7, sharex="col", sharey="row",
                    gridspec_kw={"hspace": 0, "wspace": 0}, facecolor="black",
                    figsize = (6, 4))                            
        #modify colormap
        my_cmap = plt.cm.plasma(np.arange(plt.cm.RdBu.N))
        my_cmap[:colorbar_cutoff,:4] = 0.0
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        my_cmap.set_under("w")        
        #plot
        i = 0
        for row in axs:
            for col in row:
                try:
                    img = col.imshow(np.max(atl[rngs[i]:rngs[i+1]], axis=0), cmap="gray", alpha=1)
                    img = col.imshow(np.sum(nvol[rngs[i]:rngs[i+1]], axis=0), cmap=my_cmap, alpha=0.95, vmin=0, vmax=zstep)
                    col.axis("off")
                    plt.tight_layout()
                except:
                     col.imshow(np.zeros_like(np.max(atl[rngs[0]:rngs[1]], axis=0)), cmap = "binary_r")
                     col.axis("off")  
                     plt.tight_layout()
                i += 1
        
        cbar = fig.colorbar(img, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01, ticks = [5, 10, 15, 20])
        cbar.set_label("# of planes represented",size=7)
        # access to cbar tick labels:
        cbar.ax.tick_params(labelsize=5) 
        cbar.ax.set_yticklabels(["5", "10", "15", "20"])  # vertically oriented colorbar
        
        plt.savefig(os.path.join(save_dst, "{}_neg_correlated_voxels_half_brain.pdf".format(condition)), bbox_inches = "tight", dpi=300, 
                    facecolor=fig.get_facecolor(), edgecolor="none", transparent = True)
        plt.close()
       
    return 

#%%
if __name__ == "__main__":

    #set destination of p value map you want to analyze
    atl_src = "/jukebox/LightSheetData/falkner-mouse/allen_atlas/"
    allen_id_table = os.path.join(atl_src, "allen_id_table_w_voxel_counts.xlsx")
    ann_pth = os.path.join(atl_src, "annotation_2017_25um_sagittal_forDVscans.nrrd")
    atl_pth = os.path.join(atl_src, "average_template_25_sagittal_forDVscans.tif")
    src = "/jukebox/LightSheetData/falkner-mouse/scooter/pooled_analysis/pvalue_maps"
    
    conditions = ["fm_v_mm","mf_v_mm","mf_v_fm"]
    for condition in conditions:
        make_2D_composite_overlay_of_heatmaps(os.path.join(src, "pvalues_%s.tif" % condition), 
               atl_pth, ann_pth, allen_id_table, src, condition)
                                    
    
