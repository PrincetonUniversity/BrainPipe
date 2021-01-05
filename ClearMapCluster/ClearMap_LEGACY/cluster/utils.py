#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:17:00 2016

@author: tpisano
"""

import os, sys, shutil, numpy as np, pickle
from ClearMap.cluster.preprocessing import pth_update, listdirfull, makedir, writer, removedir
from ClearMap.cluster.directorydeterminer import directorydeterminer


def update_clearmap_folder(fld):
    """Function to update clearmap folders.
    
    Inputs
    ------------
    fld = path containing output for a brain (i.e. "/home/wanglab/wang/pisano/tracing_output/sd_hsv_lob6/sd_hsv_ml150r_250u")
    """
    
    pth = "/home/wanglab/wang/pisano/Python/clearmap_cluster"
    sys.stdout.write("\nUpdating clearmap_cluster folder for {}...".format(fld)); sys.stdout.flush()
    assert os.path.exists(os.path.join(fld, "clearmap_cluster"))
    
    files = get_filepaths(pth)
    #files.remove("/home/wanglab/wang/pisano/Python/clearmap_cluster/run_clearmap_cluster.py")
    
    ##
    
    lightsheetfld = os.path.join(fld, "clearmap_cluster")
    run_tracing = [os.path.join(lightsheetfld, xx) for xx in os.listdir(lightsheetfld) if "run_clearmap_cluster.py" in xx][0]
    run_tracing_file = os.path.join(lightsheetfld [:lightsheetfld .rfind("/")], "run_clearmap_cluster.py")
    shutil.copy(run_tracing, run_tracing_file)
    sys.stdout.write("\n      run_clearmap_cluster.py copied out of clearmap fld"); sys.stdout.flush()
    shutil.rmtree(lightsheetfld)
    sys.stdout.write("\n      clearmap fld removed"); sys.stdout.flush()
    shutil.copytree(pth, lightsheetfld)
    sys.stdout.write("\n      new clearmap fld replaced"); sys.stdout.flush()

    shutil.move(run_tracing_file, lightsheetfld+"/run_clearmap_cluster.py")

    sys.stdout.write("\n      run_clearmap_cluster.py copied back into clearmap fld"); sys.stdout.flush()

    
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

               
def clean_folder(fld, deletefiles="movefiles"):
    """Function to remove or move old files before fresh re-run of steps 3-5. 
    
    Inputs
    ------------
    fld = path containing output for a brain (i.e. "/home/wanglab/wang/pisano/tracing_output/sd_hsv_lob6/sd_hsv_ml150r_250u")
    deletefiles = 
                "deletefiles" = deletes all files
                "movefiles" = moves all files into a new directory named old
    """

    assert os.path.exists(os.path.join(fld, "clearmap_cluster"))
    sys.stdout.write("\nCleaning {}...".format(fld)); sys.stdout.flush()
    ##
    allfiles = listdirfull(fld)

    #DO NOT delete good files    
    [allfiles.remove(xx) for xx in allfiles if "full_sizedatafld" in xx]
    [allfiles.remove(xx) for xx in allfiles if "clearmap_cluster" in xx and "clearmap_cluster_output" not in xx]
    [allfiles.remove(xx) for xx in allfiles if "param_dict" in xx]

    if deletefiles == "deletefiles":
        #remove files
        sys.stdout.write("\n      removing all items in remove list"); sys.stdout.flush()
        [removedir(xx) for xx in allfiles]
        sys.stdout.write("\n      all items in list removed"); sys.stdout.flush()
    elif deletefiles == "movefiles":
        #make "old" directory
        oldfld = os.path.join(fld, "old"); makedir(oldfld)
        sys.stdout.write("\n      moving all items in move list..."); sys.stdout.flush()
        for xx in allfiles:
            try:
                shutil.move(xx, oldfld)
                #copytree(xx, oldfld)
            except:
                print("\n          Moving {} Failed".format(xx))
    
    return
    
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
    return
    

def replace_lines(filepath, linenumber_text_list, verbose = False):
    """Function to replace string of text of filepath.
    
    File must exist previously. 
    Line number is ****1 based numberics****
    
    Inputs:
    -------------
    filepath: local path to clearmap_cluster package
    linenumber_textr_list: list of line number in clearmap file, and text to write in that line. e.g. lst = [[22, "I want this string to go on line 22"], [23, "this string on 23"]]
    
    """ 
    if verbose: sys.stdout.write("\nOpening file {}\n\n".format(filepath))
    
    #open file
    with open(filepath, "r") as fl:
        lines = fl.readlines()
        fl.close()

    #replace each line
    for xx in linenumber_text_list:
        lines[xx[0]-1] = xx[1]+"\n"
        if verbose: sys.stdout.write("\nReplacing line {}:\n   {}".format(xx[0], xx[1]))
        
    #rewrite file
    with open(filepath, "w") as fl:
        fl.writelines(lines)
        fl.close()

    if verbose: sys.stdout.write("\nRewriting file as {}\n\n".format(filepath))

    return

def setup_list(run_clearmap_cluster_file, *args):
    """Function to set up multiple clearmap_cluster folders.
    Must be run from within the run_clearmap_cluster package.

    Inputs
    ---------------
    run_clearmap_cluster_file : full path to run_clearmap_cluster_file.py file
    *args = list of 
    """
    #imports
    #from ClearMap.cluster.preprocessing import updateparams

    #modify run_clearmap_cluster lst
    [replace_lines(run_clearmap_cluster_file, xx, verbose = True) for xx in args]

    #load **params****
    #from run_clearmap_cluster import inputdictionary, params
        
    #run
    #updateparams(os.getcwd(), **params) # e.g. single job assuming directory_determiner function has been properly set        
    #copy folder into output for records
    #if not os.path.exists(os.path.join(params["outputdirectory"], "clearmap_cluster")): shutil.copytree(os.getcwd(), os.path.join(params["outputdirectory"], "clearmap_cluster"), ignore=shutil.ignore_patterns("^.git")) #copy run folder into output to save run info

    #del inputdictionary, params


    return
    
def load_kwargs(outdr=None, **kwargs):
    """simple function to load kwargs given an "outdr"
    
    Inputs:
    -------------
    outdr: (optional) path to folder generated by package
    kwargs
    """
    if outdr: kwargs = {}; kwargs = dict([("outputdirectory",outdr)])
    
    with open(pth_update(os.path.join(kwargs["outputdirectory"], "param_dict.p")), "rb") as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    """
    if update:
        vols = kwargs["volumes"]
        [vol.add_brainname(vol.outdr[vol.outdr.rfind("/")+1:]) for vol in vols]
        kwargs["volumes"] = vols
        
        pckloc=os.path.join(outdr, "param_dict.p"); pckfl=open(pckloc, "wb"); pickle.dump(kwargs, pckfl); pckfl.close()
    """ 
    return pth_update(kwargs)

if __name__ == "__main__":
    lst = ["/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_04_1d3x_vd_raw_largerNAsig", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_06_1d3x_vd_raw", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_03_1d3x_vd_raw", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_02_1d3x_vd_raw", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_04_1d3x_vd_raw", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_05_1d3x_vd_raw", "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_01_1d3x_vd_raw"]
    lst = ["/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_04_alignedsheets",
            "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_03_alignedsheets",
            "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_06_alignedsheets",
            "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_05_alignedsheets",
            "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_01_alignedsheets",
            "/home/wanglab/wang/pisano/tracing_output/cfos/20160531_cfos_02_alignedsheets"]
    #[clean_folder(xx, deletefiles="movefiles") for xx in lst]
    [clean_folder(xx, deletefiles="deletefiles") for xx in lst]
    [update_clearmap_folder(xx) for xx in lst]
