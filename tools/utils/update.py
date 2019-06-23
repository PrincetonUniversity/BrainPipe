#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:16:09 2017

@author: wanglab
"""
from __future__ import division
from tools.utils import *
from tools.objectdetection import find_cells
from math import ceil
import multiprocessing as mp
import numpy as np, shutil
import cPickle as pickle
import os, sys, time, warnings, collections, random
from skimage.external import tifffile
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
from tools.registration.register import make_inverse_transform, point_transform_due_to_resizing, point_transformix, transformed_pnts_to_allen
from tools.objectdetection.three_d_celldetection import detect_cells_in_3d, detect_cells_in_3d_checker
from tools.registration.transform import identify_structures_w_contours
from tools.objectdetection.injdetect import inj_detect_using_labels
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs, load_tif_list, get_filepaths
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.parallel import parallel_process
#%%
##################################################
##############UPDATE TRACING######################
##################################################    

if __name__ == '__main__':
        
    from tools.utils.update import *

    #update a single fld:
    pth = '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/l7_ts04_20150929'    
    update_lightsheet_folder(pth, updateruntracing=False)
    
    #update in parallel --UNTESTED
    #update_many_folders('/home/wanglab/wang/pisano/tracing_output/rb_n2c', cores = 2)
        
    #set pth
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    
    #just find all brains:
    allbrainpths = find_all_brains(tracing_output_fld)
    
    #run, THIS WILL TAKE A LONG TIME
    update_tracing_output(tracing_output_fld, updateruntracing=False)

if __name__ == "__main__":        
    from tools.utils.update import *    
    lst = find_all_brains('/home/wanglab/wang/pisano/tracing_output')
    
    #to update a line in run_tracing.py:
    #original_text = 'swapaxes'
    #new_text = "'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation"
    original_text ='#get job id from SLURM'    
    new_text = '    #get jobids from SLURM or argv\n    print sys.argv\n    stepid = int(sys.argv[1])    \n    if systemdirectory !="/home/wanglab/": \n        print os.environ["SLURM_ARRAY_TASK_ID"]    \n        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"]) \n    else:\n        jobid = int(sys.argv[2])'    
    for pth in lst:
        print pth
        #update_lightsheet_folder(pth)
        change_line_in_text_file(pth, original_text, new_text, number_of_lines_to_replace=5)

    
    ######change a word somewhere in package:#######################
    #set pth
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    
    #just find all brains:
    allbrainpths = find_all_brains(tracing_output_fld)
    #update from old 'tracing' package:
    [replaceword(pth, '.tracing', '.imageprocessing.preprocessing') for xx in allbrainpths for pth in listdirfull(xx) if 'param_dict' in pth]
    #revert [replaceword(pth, '.imageprocessing.preprocessing', .tracing', ) for xx in allbrainpths for pth in listdirfull(xx) if 'param_dict' in pth]

        
        
if __name__ == '__main__':
    from tools.utils import *    
    #####################################################################################
    ###use to delete everything in folder except for steps 0-2###########################
    #####################################################################################
    #set pth
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    
    #just find all brains:
    allbrainpths = find_all_brains(tracing_output_fld)
    
    #
    flds_to_clean = [xx for xx in allbrainpths if 'vc22' in xx]
    ####BEWARE********    
    #clean_all_folders(flds_to_clean, deletefiles = True)
    #############################
if __name__ == '__main__':
    from tools.utils.io import *
    from tools.utils.update import *
    #fls = listdirfull('/home/wanglab/wang/pisano/tracing_output/rb_n2c')
    fls = listdirfull('/home/wanglab/wang/pisano/tracing_output/prv/')
    #fls = listdirfull('/home/wanglab/wang/pisano/tracing_output/cfos/201701_cfos/clearmap_analysis/bkgd5_cell105_analysis/clearmap_cluster_files/')
    completedlist, unfinishedlist, nologfilelist = check_zplanes(fls, deletecompleted=False, fix_missing=False, verbose = True)

if __name__ == '__main__':
    #####################################################################################
    ###use to change atlas/annotation in runtracing files ###############################
    #####################################################################################
    #for fld in listdirfull('/home/wanglab/wang/pisano/tracing_output/zebrin', keyword='zebrin'): #*Z*ebrin
    #for fld in listdirfull('/home/wanglab/wang/pisano/tracing_output/aav'):
    #for fld in listdirfull('/home/wanglab/wang/pisano/ymaze/lightsheet_analysis/injection', keyword='cfos'): #*Z*ebrin
    for fld in listdirfull('/home/wanglab/wang/pisano/tracing_output/antero'):
        print fld
        if os.path.exists(fld+'/lightsheet'):
            original_text = "AtlasFile' : os.path.join(systemdirectory,"
            new_text = "'AtlasFile' : os.path.join(systemdirectory, 'wang/pisano/Python/atlas/sagittal_atlas_20um_iso.tif'),"
            change_line_in_run_tracing_file(fld, original_text, new_text)
            original_text = "'annotationfile'"
            new_text = "'annotationfile' : os.path.join(systemdirectory, 'wang/pisano/Python/atlas/annotation_sagittal_atlas_20um_iso.tif'), ###path to annotation file for structures"
            change_line_in_run_tracing_file(fld, original_text, new_text)
            #move
            flds = listdirfull(fld, keyword = 'elastix')
            if len(flds)>0:
                removedir(flds[-1]+'/combinedmovies')
                makedir(os.path.join(fld, 'registration_to_aba'))
                [shutil.move(xx, os.path.join(fld, 'registration_to_aba', os.path.basename(xx))) for xx in flds]
                #[shutil.rmtree(xx, os.path.join(fld, 'registration_to_aba', os.path.basename(xx))) for xx in flds]

            try:
                update_lightsheet_folder(fld, updateruntracing=False)
            except:
                pass
                
#%%
def update_many_folders(inn, cores = 2):
    '''Function to parallelize updating folders
    
    Inputs
    -----------
    inn: str to folder or list of files
    cores = for parallelization
    '''
    #return ###protection since this hasn't been tested
    if type(inn) == str: inn = listdirfull(inn)
    
    iterlst=[]; [iterlst.append((fld)) for fld in inn]
    parallel_process(iterlst, folder_helper, n_jobs=cores)
    return

def folder_helper((fld)):
    return update_lightsheet_folder(fld)

def update_lightsheet_folder(fld, updateruntracing=False):
    '''Function to update lightsheet folder.
    
    Inputs
    ------------
    fld = path containing output for a brain (i.e. '/home/wanglab/wang/pisano/tracing_output/sd_hsv_lob6/sd_hsv_ml150r_250u')
    updateruntracing: optional, if true:  for the 'run_tracing.py' it updates the file except for inputs and outdrs
    '''
    
    pth = os.path.join(directorydeterminer(), 'wang/pisano/Python/lightsheet')
    
    assert os.path.exists(os.path.join(fld, 'lightsheet'))
    
    files = get_filepaths(pth)
    files.remove(os.path.join(directorydeterminer(), 'wang/pisano/Python/lightsheet/run_tracing.py'))
    
    ##
    sys.stdout.write('\n\nFolder: {}'.format(fld)); sys.stdout.flush()
    lightsheetfld = os.path.join(fld, 'lightsheet')
    run_tracing = [os.path.join(lightsheetfld, xx) for xx in os.listdir(lightsheetfld) if 'run_tracing.py' in xx][0]
    run_tracing_file = os.path.join(lightsheetfld [:lightsheetfld .rfind('/')], 'run_tracing.py')
    shutil.copy(run_tracing, run_tracing_file)
    sys.stdout.write('\n      run_tracing.py copied out of lightsheet fld'); sys.stdout.flush()
    shutil.rmtree(lightsheetfld)
    sys.stdout.write('\n      lightsheet fld removed'); sys.stdout.flush()
    shutil.copytree(pth, lightsheetfld, ignore=shutil.ignore_patterns(*('.pyc','CVS','.git','tmp','.svn', 'TeraStitcher-Qt4-standalone-1.10.11-Linux')))
    sys.stdout.write('\n      new lightsheet fld replaced'); sys.stdout.flush()
    if updateruntracing:
        try:
            update_run_tracing(run_tracing_file, pth, lightsheetfld) ###TEMP
            sys.stdout.write('\n      run_tracing.py succcessfully updated'); sys.stdout.flush()
            shutil.move(run_tracing_file, lightsheetfld+'/backupruntracingfile.py')
        except:
            sys.stdout.write('*********FAILED: update_run_tracing, this is likely due to changes in file structure****************')
    else:
        shutil.move(run_tracing_file, lightsheetfld+'/run_tracing.py')

    sys.stdout.write('\n      run_tracing.py copied back into lightsheet fld'); sys.stdout.flush()

    
    return    
    
def update_run_tracing(run_tracing_file, pth, out):
    '''Function to update run_tracing file - note this is likely to fail with new updates, 9/14/16
    
    pth = ligthsheet folder path
    '''

    #print('check to ensure update_run_tracing function is still ok!!!!')
    
    template = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'run_tracing.py' in xx][0]
    new = []
    
    #read template    
    with open(template, 'r') as f:
        t_lines=f.readlines()
        f.close()
    #read run tracing
    with open(run_tracing_file, 'r') as f:
        r_lines=f.readlines()
        f.close()

    #template imports        
    new.append(t_lines[0:t_lines.index([xx for xx in t_lines if 'systemdirectory=preprocessing.directorydeterminer()' in xx][0])+1])
    
    #'old' inputs paths
    new.append(r_lines[r_lines.index([xx for xx in r_lines if '###set paths to data' in xx][0])-1: r_lines.index([xx for xx in r_lines if 'params={' in xx][0])+1])
    
    #new directory determiner in inputs
    new.append(t_lines[t_lines.index([xx for xx in t_lines if 'params={' in xx][0])+1: t_lines.index([xx for xx in t_lines if "inputdictionary': inputdictionary," in xx][0])])
    
    #'old' output paths+annotations
    new.append(r_lines[r_lines.index([xx for xx in r_lines if "inputdictionary': inputdictionary," in xx][0]): r_lines.index([xx for xx in r_lines if "stepid = int(sys.argv[1])" in xx][0])+1])
    
    #new function calls
    new.append(t_lines[t_lines.index([xx for xx in t_lines if "stepid = int(sys.argv[1])" in xx][0])+1:])
    
    #reformat the list
    new = [xx for x in new for xx in x]
    
    #optional to change outdr (cuz I messed up originally)
    #new[new.index([xx for xx in new if "'outputdirectory': os.path.join(systemdirectory," in xx][0])] = "    'outputdirectory': os.path.join(systemdirectory, '{}'),    \n".format(out[out.rfind('wang/pisano'):out.rfind('/')])
    
    with open(os.path.join(out, 'run_tracing.py'), "w") as filelog:
        for item in new:
            filelog.write('{}'.format(item))
        filelog.close()
    
    
    return    

def change_line_in_text_file(text_file, original_text, new_text, number_of_lines_to_replace = 0, number_to_change = None, new_save_loc=False):
    '''Function to change_line_in_run_tracing file given original_text to search for a new_text to replace.
    
    NOTE:WILL CHANGE FOR ALL LINES FOUND...UNLESS YOU PROVIDE number_to_change
    
    Inputs
    -----------------
    text_file: path to text_file.txt
    original_text: str to search for in that line and that is unique to that line
        e.g.:
            original_text = 'swapaxes'
    new_text: new text that will REPLACE old line
        e.g.:
            new_text = "'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation"
    number_to_change (optional):
        int: if multiple copies of original_text are present and you only want to change a single line, then use this to select which one. NOTE USES ZERO-BASED NUMBERING
    number_of_lines_to_replace (optional):
        int: number of lines AFTER the original_text to remove. I.e. to remove the line with the original text and the next then use 2.
    new_save_loc (optional): if true, new file name to save, if false save over old file
    '''
    
    if not number_of_lines_to_replace: number_of_lines_to_replace = 0
    
    with open(text_file, 'r') as f:
        lines=f.readlines()
        f.close()   

    #find text and replace string.
    if original_text:
        if not number_to_change:
            for indices in [lines.index(xx) for xx in lines if original_text in xx]:
                del lines[indices: indices+number_of_lines_to_replace]
                lines[indices] = new_text+'\n'
                
        else:
            for indices in [[lines.index(xx) for xx in lines if original_text in xx][number_to_change]]:
                del lines[indices: indices+number_of_lines_to_replace]
                lines[indices] = new_text+'\n'
    
    #or find lines and replace string # NOT FUNCTIONAL
    elif lines_to_replace:
        del lines[lines_to_replace[0]:lines_to_replace[1]]
        lines.insert(lines_to_replace[0], new_text+'\n')

    if new_save_loc == False: new_save_loc = text_file
    
    #save output
    with open(new_save_loc, "w") as filelog:
        for item in lines:
            filelog.write('{}'.format(item))
        filelog.close()    
    
    return

def change_line_in_run_tracing_file(pth_to_lightsheet_folder, original_text, new_text, number_of_lines_to_replace = None, number_to_change = None, new_save_loc=False):
    '''Function to change_line_in_text_file given original_text to search for a new_text to replace.
    
    NOTE:WILL CHANGE FOR ALL LINES FOUND...UNLESS YOU PROVIDE number_to_change
    
    Inputs
    -----------------
    original_text: str to search for in that line and that is unique to that line
        e.g.:
            original_text = 'swapaxes'
    new_text: new text that will REPLACE old line
        e.g.:
            new_text = "'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation"
    number_to_change (optional):
        int: if multiple copies of original_text are present and you only want to change a single line, then use this to select which one. NOTE USES ZERO-BASED NUMBERING
    number_of_lines_to_replace (optional): 0 BASED NUMERICS
        int: number of lines after the original_text to remove. I.e. to remove the line with the original text and the next then use 2.
    new_save_loc (optional): if true, new file name to save, if false save over old file
    '''
    
    if not number_of_lines_to_replace: number_of_lines_to_replace = 0
    
    if pth_to_lightsheet_folder[-10:] != 'lightsheet': pth_to_lightsheet_folder = os.path.join(pth_to_lightsheet_folder, 'lightsheet')
    run_tracing_file = [os.path.join(pth_to_lightsheet_folder, xx) for xx in os.listdir(pth_to_lightsheet_folder) if 'run_tracing.py' in xx][0]
    
    with open(run_tracing_file, 'r') as f:
        lines=f.readlines()
        f.close()   

    #find text and replace string.
    if original_text:
        if not number_to_change:
            for indices in [lines.index(xx) for xx in lines if original_text in xx]:
                del lines[indices: indices+number_of_lines_to_replace]
                lines[indices] = new_text+'\n'
                
        else:
            for indices in [[lines.index(xx) for xx in lines if original_text in xx][number_to_change]]:
                del lines[indices: indices+number_of_lines_to_replace]
                lines[indices] = new_text+'\n'
    
    #or find lines and replace string
    elif lines_to_replace:
        del lines[lines_to_replace[0]:lines_to_replace[1]]
        lines.insert(lines_to_replace[0], new_text+'\n')

    if new_save_loc == False: new_save_loc = run_tracing_file
    
    #save output
    with open(new_save_loc, "w") as filelog:
        for item in lines:
            filelog.write('{}'.format(item))
        filelog.close()    
    
    return
    

    
def search_and_replace_textfile(txtfl, original_text, new_text, new_save_loc=False, verbose=False):
    '''Function to search and replace strings
    
    NOTE:WILL CHANGE FOR ALL LINES FOUND...UNLESS YOU PROVIDE number_to_change
    
    Inputs
    -----------------
    original_text: str to search for in that line and that is unique to that line
        e.g.:
            original_text = 'swapaxes'
    new_text: new text that will REPLACE old line
        e.g.:
            new_text = "'finalorientation' :  ('2','1','0'), #Used to account for different orientation between brain and atlas. Assumes XYZ ('0','1','2) orientation. Pass strings NOT ints. '-0' = reverse the order of the xaxis. For better description see docstring from tools.imageprocessing.orientation.fix_orientation"
    new_save_loc (optional): if true, new file name to save, if false save over old file
    '''
    
       
    with open(txtfl, 'r') as f:
        lines=f.readlines()
        f.close()   

    if new_save_loc == False: new_save_loc = txtfl
    
    #save output
    f1 = open(new_save_loc, "w")
    
    #find text and replace string.
    for line in lines:
        f1.write(line.replace(original_text, new_text))
        if verbose: print line
    
   
    return
    

def update_tracing_output(tracing_output, updateruntracing=False):
    '''Wrapper to handle the I/O of update_lightsheet_folder function
    
    Input
    ------------------
    tracing_output: str(master folder) --or-- list of folders to update
    updateruntracing: optional, if true:  for the 'run_tracing.py' it updates the file except for inputs and outdrs
    '''
    
    #make flds list
    flds = []
    
    if type(tracing_output) == str:
    
        #really ugly way to walk through all of tracing_output and find brain flds
        try:
            for x in os.listdir(tracing_output):
                try:
                    for xx in os.listdir(os.path.join(tracing_output, x)):
                        try:
                            if 'lightsheet' in os.listdir(os.path.join(tracing_output, x, xx)):
                                flds.append(os.path.join(tracing_output, x, xx))
                        except:
                            pass
                except:
                    pass
        except:
            pass
    elif type(tracing_output) == list:
        flds = tracing_output
        
    sys.stdout.write('Updating {} folders, this will take ~3 minutes/fld...\n'.format(len(flds))); sys.stdout.flush()
    tick = 0
    for fld in flds:
        sys.stdout.write('\nUpdating {}...'.format(fld))
        update_lightsheet_folder(fld, updateruntracing = updateruntracing)
        sys.stdout.write('\n ...completed, {} of {}'.format(tick+1, len(flds)))
        tick+=1
    
    return
    
def find_all_brains(tracing_output):
    #make flds list
    flds = []
    
    #really ugly way to walk through all of tracing_output and find brain flds
    try:
        for x in os.listdir(tracing_output):
            try:
                for xx in os.listdir(os.path.join(tracing_output, x)):
                    try:
                        if 'lightsheet' in os.listdir(os.path.join(tracing_output, x, xx)):
                            flds.append(os.path.join(tracing_output, x, xx))
                    except:
                        pass
            except:
                pass
    except:
        pass
    return flds
    
    
#%%
def clean_all_folders(tracing_output, deletefiles = False):
    '''Function to remove or move old files before fresh re-run of steps 3-5
    
    Input
    ------------------
    tracing_output: str(master folder) --or-- list of folders to update
    deletefiles = 
                True = deletes all files
                False = moves all files into a new directory named old
    '''
    
    #make flds list
    flds = []
    
    if type(tracing_output) == str:
    
        #really ugly way to walk through all of tracing_output and find brain flds
        try:
            for x in os.listdir(tracing_output):
                try:
                    for xx in os.listdir(os.path.join(tracing_output, x)):
                        try:
                            if 'lightsheet' in os.listdir(os.path.join(tracing_output, x, xx)):
                                flds.append(os.path.join(tracing_output, x, xx))
                        except:
                            pass
                except:
                    pass
        except:
            pass
    elif type(tracing_output) == list:
        flds = tracing_output
        
    sys.stdout.write('Cleaning of tracing_output, this will take 2 minutes/folder...\n')
    tick = 0
    for fld in flds:
        clean_folder(fld, deletefiles)
        sys.stdout.write('\n ...completed, {} of {}'.format(tick+1, len(flds))); sys.stdout.flush()
        tick+=1
    
    return
    
def clean_folder(fld, deletefiles='movefiles', removetifs = False, removerawdata = False):
    '''Function to remove or move old files before fresh re-run of steps 3-5. 
    
    Inputs
    ------------
    fld = path containing output for a brain (i.e. '/home/wanglab/wang/pisano/tracing_output/sd_hsv_lob6/sd_hsv_ml150r_250u')
    deletefiles = 
                'deletefiles' = deletes all files
                'movefiles' = moves all files into a new directory named old
    removetifs (optional) = 
                True: remove downsized tiffs in folder
                False: does not remove
    removerawdata (optional) = 
                True: remove rawdata in full_sizedatafld
                False: does not remove
    '''

    assert os.path.exists(os.path.join(fld, 'lightsheet'))
    sys.stdout.write('\nCleaning {}...'.format(fld)); sys.stdout.flush()
    ##
    allfiles = listdirfull(fld)

    #    
    removelst=[]
    [removelst.append(xx) for xx in allfiles if os.path.join(fld, 'elastix') in xx]
    [removelst.append(xx) for xx in allfiles if os.path.join(fld, 'transformedpoints_pretransformix') in xx]
    [removelst.append(xx) for xx in allfiles if '.xlsx' in xx]
    [removelst.append(xx) for xx in allfiles if 'structure_density' in xx]
    [removelst.append(xx) for xx in allfiles if os.path.join(fld, 'injection') in xx]
    [removelst.append(xx) for xx in allfiles if os.path.join(fld, 'cells') in xx]
    [removelst.append(xx) for xx in allfiles if '3D_contours' in xx]
    [removelst.append(xx) for xx in allfiles if 'step4_out.txt' in xx]
    #added protection against deleting good files    
    [removelst.remove(xx) for xx in removelst if os.path.join(fld, 'full_sizedatafld') in xx]
    [removelst.remove(xx) for xx in removelst if os.path.join(fld, 'lightsheet') in xx]
    [removelst.remove(xx) for xx in removelst if 'param_dict' in xx]

    #optionals:
    if removetifs: [removelst.append(xx) for xx in allfiles if '.tif' in xx]
    if removerawdata: [removelst.append(xx) for xx in allfiles if 'full_sizedatafld' in xx]
     
    if deletefiles == 'deletefiles':
        #remove files
        sys.stdout.write('\n      removing all items in remove list'); sys.stdout.flush()
        [removedir(xx) for xx in removelst]
        sys.stdout.write('\n      all items in list removed'); sys.stdout.flush()
    elif deletefiles == 'movefiles':
        #make 'old' directory
        oldfld = os.path.join(fld, 'old'); makedir(oldfld)
        sys.stdout.write('\n      moving all items in move list...'); sys.stdout.flush()
        for xx in removelst:
            try:
                shutil.move(xx, oldfld)
                #copytree(xx, oldfld)
            except:
                print('\n          Moving {} Failed'.format(xx))
    sys.stdout.write('\n      replacing folder structure...'); sys.stdout.flush()           
    #replace folders
    makedir(os.path.join(fld, 'injection'))
    makedir(os.path.join(fld, 'injection', 'injcoordinatesfld'))
    makedir(os.path.join(fld, 'injection', 'injdetect3d'))
    makedir(os.path.join(fld, 'cells'))    
    makedir(os.path.join(fld, 'cells', 'celldetect3d'))    
    makedir(os.path.join(fld, 'cells', 'cellcoordinatesfld'))    
    
    sys.stdout.write('\n      new injection + cell flds replaced\n\n\n')
    
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

def check_zplanes(args, deletecompleted=False, fix_missing=False, verbose = True):
    '''check to make sure all zplanes (step1) have completed

    
    Inputs:
        ----------
    args: list of lightsheet folders each consisting of a single full job with a LogFile present
    
    args = listdirfull('/home/wanglab/wang/pisano/tracing_output/antero')
    fix_missing (optional): attempt to fix missing zplanes
    
    Returns:
        ---------
    completedlist, unfinishedlist, nologfilelist
    '''
    import os, sys, shutil
    from tools.utils.io import load_kwargs, listdirfull
    completedlist=[]; unfinishedlist=[]; nologfilelist=[]
    for pth in args:
        if verbose: sys.stdout.write('\n{}\n  '.format(pth)); sys.stdout.flush()
        kwargs = load_kwargs(pth)
        try:
            #with open(kwargs['outputdirectory']+'/LogFile.txt','r') as f:
            #    lines = f.read()
            #    f.close()
            #if len(kwargs['volumes']) == lines.count('STEP 1: Correct numbe'): 
            if np.all([len(listdirfull(vol.full_sizedatafld_vol)) == vol.fullsizedimensions[0] for vol in kwargs['volumes']]):
                [completedlist.append(xx) for xx in kwargs['inputdictionary'].keys()]
                if verbose: sys.stdout.write('   complete'); sys.stdout.flush()
            else:
                [unfinishedlist.append(xx) for xx in kwargs['inputdictionary'].keys()]
                if fix_missing:
                    from tools.imageprocessing.preprocessing import process_planes_completion_checker
                    process_planes_completion_checker(**kwargs)
                if verbose: sys.stdout.write('   INCOMPLETE'); sys.stdout.flush()
        except:
            nologfilelist.append(pth)
            if verbose: sys.stdout.write('   error of some sort - log file found'); sys.stdout.flush()
            
    if deletecompleted: [shutil.rmtree(xx) for xx in completedlist if xx in listdirfull(xx[:xx.rfind('/')])]
                                      
    return completedlist, unfinishedlist, nologfilelist
        