#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:14:48 2017

@author: tpisano

"""

import os, numpy as np
import tifffile
import matplotlib as mpl
from tools.imageprocessing.orientation import fix_orientation
from tools.registration.transform import count_structure_lister, transformed_pnts_to_allen_helper_func
from tools.utils.io import load_kwargs, makedir
from collections import Counter
import SimpleITK as sitk, pandas as pd
import matplotlib.pyplot as plt; plt.ion()

def orientation_crop_check(src, axes = ("0","1","2"), crop = False, dst=False):
    """Function to check orientation and cropping. MaxIPs along 0 axis.
    
      "crop": #use to crop volume, values below assume horizontal imaging and sagittal atlas
                False
                cerebellum: "[:,390:,:]"
                caudal midbrain: "[:,300:415,:]"
                midbrain: "[:,215:415,:]"
                thalamus: "[:,215:345,:]"
                anterior cortex: "[:,:250,:]"
    "dst": (optional) path+extension to save image
                
    Returns
    ---------------
    cropped image
    
    """
    fig = plt.figure()
    plt.axis("off")
    ax = fig.add_subplot(1,2,1)
    if type(src) == str: src = tifffile.imread(src)
    plt.imshow(np.max(src, axis=0))
    plt.title("Before reorientation")
    
    ax = fig.add_subplot(1,2,2)
    if crop: src = eval("src{}".format(crop))
    src = fix_orientation(src, axes=axes)
    plt.imshow(np.max(src, axis=0))
    plt.title("After reorientation")
    
    if dst: plt.savefig(dst, dpi=300)
    return src

def optimize_inj_detect(src, threshold=3, filter_kernel = (3,3,3), dst=False):
    """Function to test detection parameters
    
    "dst": (optional) path+extension to save image
    
    """
    if type(src) == str: src = tifffile.imread(src)
    arr = find_site(src, thresh=threshold, filter_kernel=filter_kernel)*45000
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(np.max(arr, axis=0));  plt.axis("off")
    fig.add_subplot(1,2,2)
    plt.imshow(np.max(src, axis=0), cmap="jet");  plt.axis("off")
    
    if dst: plt.savefig(dst, dpi=300)
    
    return 

def pool_injections_for_analysis(**kwargs):
    """Function to pool several injection sites. Assumes that the basic registration using this software has been run.
       
    Inputs
    -----------
    kwargs:
      "inputlist": inputlist, #list of folders generated previously from software
      "inputtype": "main_folder", "tiff" #specify the type of input. main_folder is the lightsheetpackage"s folder, tiff is the file 
      to use for injection site segmentation
      "channel": "01", 
      "channel_type": "injch",
      "filter_kernel": (3,3,3), #gaussian blur in pixels (if registered to ABA then 1px likely is 25um)
      "threshold": 3 (int, value to use for thresholding, this value represents the number of stand devs above the mean of the gblurred
      image)
      "num_sites_to_keep": int, number of injection sites to keep, useful if multiple distinct sites
      "injectionscale": 45000, #use to increase intensity of injection site visualizations generated - DOES NOT AFFECT DATA
      "imagescale": 2, #use to increase intensity of background  site visualizations generated - DOES NOT AFFECT DATA
      "reorientation": ("2","0","1"), #use to change image orientation for visualization only
      "crop": #use to crop volume, values below assume horizontal imaging and sagittal atlas
                False
                cerebellum: "[:,390:,:]"
                caudal midbrain: "[:,300:415,:]"
                midbrain: "[:,215:415,:]"
                thalamus: "[:,215:345,:]"
                anterior cortex: "[:,:250,:]"
      
      "dst": "/home/wanglab/Downloads/test", #save location
      "save_individual": True, #optional to save individual images, useful to inspect brains, which you can then remove bad brains from 
      list and rerun function
      "colormap": "plasma", 
      "atlas": "/jukebox/wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif",
      "annotation":"/jukebox/wang/pisano/Python/allenatlas/annotation_25_ccf2015_forDVscans.nrrd",
      "id_table": "/jukebox/temp_wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx",
      
      Optional:
          ----------
          "save_array": path to folder to save out numpy array per brain of binarized detected site
          "save_tif": saves out tif volume per brain of binarized detected site
          "dpi": dots per square inch to save at
          "crop_atlas":(notfunctional) similiar to crop. Use when you would like to greatly restrain the cropping for injsite detection,
          but you want to display a larger area of overlay.
                      this will 0 pad the injection sites to accomodate the difference in size. Note this MUST be LARGER THAN crop.
          
      Returns
      ----------------count_threshold
      a pooled image consisting of max IP of reorientations provide in kwargs.
      a list of structures (csv file) with pixel counts, pooling across brains.
      if save individual will save individual images, useful for inspection and/or visualization
    """
    
    inputlist = kwargs["inputlist"]
    inputtype = kwargs["inputtype"] if "inputtype" in kwargs else "main_folder"
    dst = kwargs["dst"]; makedir(dst)
    injscale = kwargs["injectionscale"] if "injectionscale" in kwargs else 1
    imagescale = kwargs["imagescale"] if "imagescale" in kwargs else 1
    axes = kwargs["reorientation"] if "reorientation" in kwargs else ("0","1","2")
    cmap = kwargs["colormap"] if "colormap" in kwargs else "plasma"
    id_table = kwargs["id_table"] if "id_table" in kwargs else "/jukebox/LightSheetTransfer/atlas/ls_id_table_w_voxelcounts.xlsx"
    save_array = kwargs["save_array"] if "save_array" in kwargs else False
    save_tif = kwargs["save_tif"] if "save_tif" in kwargs else False
    num_sites_to_keep = kwargs["num_sites_to_keep"] if "num_sites_to_keep" in kwargs else 1
    nonzeros = []
    ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs["annotation"]))
    if kwargs["crop"]: ann = eval("ann{}".format(kwargs["crop"]))   
    allen_id_table=pd.read_excel(id_table)
    
    
    for i in range(len(inputlist)):
        pth = inputlist[i]
        print("\n\n_______\n{}".format(os.path.basename(pth)))
        #find the volume by loading the param dictionary generated using the light-sheet package
        if inputtype == "main_folder":
            dct = load_kwargs(pth); #print dct["AtlasFile"]
            vol = [xx for xx in dct["volumes"] if xx.ch_type in kwargs["channel_type"] and xx.channel == kwargs["channel"]][0]
            #done to account for different versions
            if os.path.exists(vol.ch_to_reg_to_atlas+"/result.1.tif"):
                impth = vol.ch_to_reg_to_atlas+"/result.1.tif"
            elif os.path.exists(vol.ch_to_reg_to_atlas) and vol.ch_to_reg_to_atlas[-4:]==".tif":
                impth = vol.ch_to_reg_to_atlas
            elif os.path.exists(os.path.dirname(vol.ch_to_reg_to_atlas)+"/result.1.tif"):
                impth = os.path.dirname(vol.ch_to_reg_to_atlas)+"/result.1.tif"
            elif os.path.exists(os.path.dirname(vol.ch_to_reg_to_atlas)+"/result.tif"):
                impth = os.path.dirname(vol.ch_to_reg_to_atlas)+"/result.tif"
        else: #"tiff", use the file given
            impth = pth
        
        print("  loading:\n     {}".format(pth))
        im = tifffile.imread(impth)
            
        if kwargs["crop"]: im = eval("im{}".format(kwargs["crop"]))#; print im.shape
        
        #segment
        arr = find_site(im, thresh=kwargs["threshold"], filter_kernel=kwargs["filter_kernel"], 
                        num_sites_to_keep=num_sites_to_keep)*injscale
        if save_array: np.save(os.path.join(save_array,"{}".format(os.path.basename(pth))+".npy"), arr.astype("float32"))
        if save_tif: tifffile.imsave(os.path.join(save_tif,"{}".format(os.path.basename(pth))+".tif"), arr.astype("float32"))
        
        #optional "save_individual"
        if kwargs["save_individual"]:
            im = im*imagescale
            a=np.concatenate((np.max(im, axis=0), np.max(arr.astype("uint16"), axis=0)), axis=1)
            b=np.concatenate((np.fliplr(np.rot90(np.max(fix_orientation(im, axes=axes), axis=0),k=3)), 
                              np.fliplr(np.rot90(np.max(fix_orientation(arr.astype("uint16"), axes=axes), axis=0),k=3))), axis=1)
            plt.figure()
            plt.imshow(np.concatenate((b,a), axis=0), cmap=cmap, alpha=1);  plt.axis("off")
            plt.savefig(os.path.join(dst,"{}".format(os.path.basename(pth))+".pdf"), dpi=300, transparent=True)
            plt.close()

        #cell counts to csv
        print("   finding nonzero pixels for voxel counts...")      
        nz = np.nonzero(arr)
        nonzeros.append(list(zip(*nz))) #<-for pooled image
        pos = transformed_pnts_to_allen_helper_func(np.asarray(list(zip(*[nz[2], nz[1], nz[0]]))), ann)
        tdf = count_structure_lister(allen_id_table, *pos)
        if i == 0: 
            df = tdf.copy()
            countcol = "count" if "count" in df.columns else "cell_count"
            df.drop([countcol], axis=1, inplace=True)
        df[os.path.basename(pth)] = tdf[countcol]
        
    df.to_csv(os.path.join(dst,"voxel_counts.csv"), index = False)
    print("\n\nCSV file of cell counts, saved as {}\n\n\n".format(os.path.join(dst,"voxel_counts.csv")))  
            
    #condense nonzero pixels
    nzs = [str(x) for xx in nonzeros for x in xx] #this list has duplicates if two brains had the same voxel w label
    c = Counter(nzs)
    array = np.zeros(im.shape)
    print("Collecting nonzero pixels for pooled image...")
    tick = 0
    #generating pooled array where voxel value = total number of brains with that voxel as positive
    for k,v in c.items():
        k = [int(xx) for xx in k.replace("(","").replace(")","").split(",")]
        array[k[0], k[1], k[2]] = int(v)
        tick+=1
        if tick % 50000 == 0: print("   {}".format(tick))
        
    #load atlas and generate final figure
    print("Generating final figure...")      
    atlas = tifffile.imread(kwargs["atlas"])
    arr = fix_orientation(array, axes=axes)
    #cropping
    if kwargs["crop"]: atlas = eval("atlas{}".format(kwargs["crop"]))
    atlas = fix_orientation(atlas, axes=axes)

    my_cmap = eval("plt.cm.{}(np.arange(plt.cm.RdBu.N))".format(cmap))
    my_cmap[:1,:4] = 0.0  
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    my_cmap.set_under("w")
    plt.figure()
    plt.imshow(np.max(atlas, axis=0), cmap="gray")
    plt.imshow(np.max(arr, axis=0), alpha=0.99, cmap=my_cmap); plt.colorbar(); plt.axis("off")
    dpi = int(kwargs["dpi"]) if "dpi" in kwargs else 300
    plt.savefig(os.path.join(dst,"heatmap.pdf"), dpi=dpi, transparent=True);
    plt.close()
    
    print("Saved as {}".format(os.path.join(dst,"heatmap.pdf")))  
        
    return df


def find_site(im, thresh=10, filter_kernel=(5,5,5), num_sites_to_keep=1):
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
    if type(im) == str: im = tifffile.imread(im)

    filtered = gfilt(im, filter_kernel)
    thresholded = filtered > filtered.mean() + thresh*filtered.std() 
    labelled,nlab = label(thresholded)

    if nlab == 0:
        raise Exception("Site not detected, try a lower threshold?")
    elif nlab == 1:
        return labelled.astype(bool)
    elif num_sites_to_keep == 1:
        sizes = [np.sum(labelled==i) for i in range(1,nlab+1)]
        return labelled == np.argmax(sizes)+1
    else:
        sizes = [np.sum(labelled==i) for i in range(1,nlab+1)]
        vals = [i+1 for i in np.argsort(sizes)[-num_sites_to_keep:][::-1]]
        return np.in1d(labelled, vals).reshape(labelled.shape)
    
if __name__ == "__main__":
    
    """ NOTE: the default functions in this scripts assumes you have a 'processed' lightsheet directory for each brain; if not, 
    you will need to modify accordingly """
    
    #check if reorientation is necessary
    src = "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an10/elastix/an10_devcno_03082019_1d3x_488_017na_1hfds_z10um_100msec_resized_ch00/result.tif"
    src = orientation_crop_check(src, axes = ("2","0","1"), crop = "[:,423:,:]")
    
    #optimize detection parameters for inj det
    optimize_inj_detect(src, threshold=3, filter_kernel = (5,5,5))
    
    #run
    #suggestion: save_individual=True,
    #then inspect individual brains, which you can then remove bad brains from list and rerun function
    inputlist = [#"/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an25",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an16",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an04", #weird registration but probably ok
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an17",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an07",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an03",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an14",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an02",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an26",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an23",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an31",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an30",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an09",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an06",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an22",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an12",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an27",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an15",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an10",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an20",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an21",
#                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an24",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an01",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an13",
                 "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/processed/an05"]

    kwargs = {"inputlist": inputlist,
              "inputtype": "main_folder", 
              "channel_type": ["injch", "cellch"], #must be a list, safer as this can be mislabeled
              "channel": "00",
              "filter_kernel": (5,5,5),
              "threshold": 3,
              "num_sites_to_keep": 1,
              "injectionscale": 45000, 
              "imagescale": 2,
              "reorientation": ("2","0","1"),
              "crop": "[:,423:,:]",
              "dst": "/jukebox/wang/Jess/lightsheet_output/201906_development_cno/pooled_analysis",
              "save_individual": True, 
              "colormap": "plasma", 
              "atlas": "/jukebox/LightSheetTransfer/atlas/sagittal_atlas_20um_iso.tif",
              "annotation":"/jukebox/LightSheetTransfer/atlas/annotation_sagittal_atlas_20um_iso.tif",
              "id_table": "/jukebox/LightSheetTransfer/atlas/ls_id_table_w_voxelcounts.xlsx"
            }              
              
    
    df = pool_injections_for_analysis(**kwargs)