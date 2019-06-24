# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:05:13 2016

@author: wanglab
"""
####file from: http://help.brain-map.org/display/api/Downloading+an+Ontology's+Structure+Graph
###download: http://api.brain-map.org/api/v2/structure_graph_download/1.json

import pandas as pd, os, numpy as np
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer
from tools.registration.register import allen_compare
from skimage.external import tifffile
import SimpleITK as sitk
from collections import Counter


def allen_structure_json_to_pandas(pth, prune=True, ann_pth = None, save=False):    
    '''simple function to pull; this version prunes and cleans more
    ****prune should equal true for compatibility with other functions in package****
    
    pth='/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.json'
    
    '''
    if save == False: save = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx' 
    ##preserves the structures from Json file    
    if prune == False:    
        ###load file and chunk each structure into a list item    
        with open(pth, 'r') as f:
            lines=f.readlines()
        lines=lines[7:] ####remove the header junk, numbers may need to be changed with newer structures
        bg_lst=[]    
        for i in range(len(lines)):
            if '"id":' in lines[i]:
                tmp=[lines[x][lines[x].find('"'):lines[x].rfind('\n')] for x in range(i, i+11)]
                bg_lst.append(tmp)
        pruned_lst=[]
        for i in bg_lst:
            inputs=[str(ii[ii.rfind(':')+2:ii.rfind(',')]) for ii in i]
            inputs[3:6]=[x[1:-1] for x in inputs[3:6]]
            headers=[str(ii[ii.find('"')+1:ii.rfind(':')-1]) for ii in i]
            pruned_lst.append(inputs)
        #### put into a pandas table
        #####ID=pixel value for annotated file
        df=pd.DataFrame(pruned_lst, columns=headers)
        df.to_excel(save)
    
    ###*******PRUNE****makes working with the files MUCH easier
    else:
        ###load file and chunk each structure into a list item    
        with open(pth, 'r') as f:
            lines=f.readlines()
        lines=lines[7:] ####remove the header junk, numbers may need to be changed with newer structures
        bg_lst=[]    
        for i in range(len(lines)):
            if '"id":' in lines[i]:
                tmp=[lines[x][lines[x].find('"'):lines[x].rfind('\n')] for x in [i+4, i+3, i, i+1, i+9]]
                bg_lst.append(tmp)
                
        pruned_lst=[]
        for i in bg_lst:
            inputs=[str(ii[ii.rfind(':')+2:ii.rfind(',')]) for ii in i]
            inputs[0:2]=[x[1:-1] for x in inputs[0:2]]
            headers=[str(ii[ii.find('"')+1:ii.rfind(':')-1]) for ii in i]; [headers.append(x) for x in ['parent_acronym', 'parent_name']]
            pruned_lst.append(inputs)
    
        ##adding parent acronym and parent name    
        lst2=[]
        for i in range(len(pruned_lst)):
            if pruned_lst[i][-1] != 'null' and pruned_lst[i][-2] != 'null': ##second parts gets rid of pixel labeled ids
                tmp=[(x[0], x[1]) for x in pruned_lst if pruned_lst[i][-1] == x[2]]
                tmp2=[x for x in pruned_lst[i]]; 
                if len(tmp) !=0:
                    [tmp2.append(x) for x in tmp[0]]
                else:
                    [tmp2.append(x) for x in ['null', 'null']]
                lst2.append(tmp2)
            else:
                tmp2=[x for x in pruned_lst[i]]; [tmp2.append(x) for x in ['null', 'null']]
                lst2.append(tmp2)
        '''
        #optional add MISSING info: NOTE NOT NECESSARILY VENTRICLES
        #Index([u'name', u'acronym', u'id', u'atlas_id', u'parent_structure_id',u'parent_acronym', u'parent_name'] dtype='object')
        #if add_ventricles:
            lst2.append(['Ventricles', 'Vent', '2222','2222','997', 'root', 'root'])
            lst2.append(['Lateral Ventricles', 'LV', '2','2','2222', 'Vent', 'Ventricles']) #pix id of 2 on atlas
            other missing:
                10703.0
                10704.0
                182305696.0
                182305712.0
                312782560.0
                312782592.0
                312782624.0
                312782656.0
                #atl = tifffile.imread('/jukebox/wang/pisano/Python/allenatlas/average_template_25_sagittal_forDVscans.tif')
                #tann = np.zeros(ann.shape); tann[np.where(ann==10703)] = 255
                #sitk.Show(sitk.GetImageFromArray(tann.astype('uint8'))); sitk.Show(sitk.GetImageFromArray(atl.astype('uint8')))
                
           ''' 
        #### put into a pandas table
        #####ID=pixel value for annotated file
        df=pd.DataFrame(lst2, columns=headers)
        df.to_excel(save)
#%%

def isolate_structures(pth, *args):
    '''helper function to segment out allen brain structures.
    Inputs:
        pth=pth to allen annoation
        *args=lst of structure ID's (pixel values) to keep
    '''
    im = sitk.GetArrayFromImage(sitk.ReadImage(pth))
    res = np.zeros(im.shape).astype('uint8')
    for l in args:
        res[im==l] = 255
    return res

def isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv,*args):
    '''function to segment out allen brain structures and then depth color overlay.
    Inputs:
        pth=pth to allen annoation
        AtlasFile=pth to atlas (grayscale)
        svlc=pth to folder to save
        nm_to_sv=text to identify this structure
        *args=lst of structure ID's (pixel values) to keep
    '''
    im=isolate_structures(pth, *args)
    makedir(svlc)
    svlc2=os.path.join(svlc, nm_to_sv)
    makedir(svlc2)    
    impth=os.path.join(svlc2, str(nm_to_sv)+'.tif')    
    tifffile.imsave(impth, im.astype('uint8'))
    allen_compare(AtlasFile, svlc, impth)
    return

def overlay(AtlasFile, svlc, nm_to_sv, arr, gaussianblur=False):
    '''function to plot coordinates using depth color overlay.
    Inputs:
        pth=pth to allen annoation
        AtlasFile=pth to atlas (grayscale)
        svlc=pth to folder to save
        nm_to_sv=text to identify this structure
        arr= np array of [zyx] PIXEL COORDINATES to plot
        gaussianblur (optional):False/True
    '''
    im = sitk.GetArrayFromImage(sitk.ReadImage(AtlasFile))
    im = np.zeros(im.shape).astype('uint8')
    for i in tuple(map(tuple, arr.astype('int'))):
        im[i] = 255
    #optional
    if gaussianblur:
        from scipy.ndimage.filters import gaussian_filter
        im = gaussian_filter(im, (2,2,2)).astype('uint8')
        im[im>0]=255
        im = gaussian_filter(im, (2,2,2)).astype('uint8')
        im[im>0]=255
    makedir(svlc)
    svlc2=os.path.join(svlc, nm_to_sv)
    makedir(svlc2)    
    impth=os.path.join(svlc2, str(nm_to_sv)+'.tif')    
    tifffile.imsave(impth, im.astype('uint8'))
    allen_compare(AtlasFile, svlc, impth)
    return
    
def annotation_value_to_structure(allen_id_table, args):
    '''Function that returns a list of structures based on annotation pixel value
    
    Removes 0 from list
    
    Inputs:
        allen_id_table=path to excel file generated from scripts above
        *args=list of allen ID pixel values
    '''
    df = pd.read_excel(allen_id_table)
    
    if type(args) != list:
        return str(list(df.name[df.id==args])[0])
    else:
        args = [int(i) for i in args if int(i) != 0]
        print([i for i in args if i not in df.id])
        return [(i, str(list(df.name[df.id==i])[0])) for i in args if i in df.id]

def annotation_location_to_structure(id_table, args, ann=False):
    '''Function that returns a list of structures based on annotation z,y,x cooridnate
    
    Removes 0 from list
    
    Inputs:
        id_table=path to excel file generated from scripts above
        args=list of ZYX coordinates. [[77, 88, 99], [12,32,53]]
        ann = annotation file
        
    ****BE AWARE OF ORIENTATION CHANGES and if cropping crop both vol and ann****
        
    Returns
    ---------
    list of counts, structure name
    '''
    df = pd.read_excel(id_table)
    #if not str(type(ann)) == 'numpy.ndarray':
    #    print('loading ann as /jukebox/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd')
    #    ann = sitk.GetArrayFromImage(sitk.ReadImage('/jukebox/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'))
    
    #find locs 
    vals=[]; [vals.append(ann[i[0], i[1], i[2]]) for i in args]
    c = Counter(vals) #dict of value: occurences
        
    #remove 0 and find structs
    if 0 in c: del c[0]
    
    #some of the values aren't annotated properly?
    lst = []
    for k,v in c.iteritems():
        try:
            lst.append((v, str(list(df.name[df.id==k])[0])))
        except Exception as e:
            print('Removing {}, as generating this error: {}'.format(k, e))

        
    #[(v, str(list(df.name[df.id==k])[0])) for k,v in c.iteritems()]
    return lst
                

def isolate_structures_return_list(allen_id_table, ann, args, bitdepth='uint16', verbose=False):
    '''Function that generates evenly spaced pixels values based on annotation 
    
    Removes 0 from list
    
    Inputs:
        allen_id_table=path to excel file generated from scripts above
        ann = allen annoation file
        args=list of allen ID pixel values*********************
        
    Returns:
        -----------
        nann = new array of bitdepth
        list of value+name combinations
    '''
    if type(ann) == str: ann = sitk.GetArrayFromImage(sitk.ReadImage(ann))

    df = pd.read_excel(allen_id_table)
    args = [float(i) for i in args if i != 0]
            
    #remove pix values not represented in atlas...determined emperically; why IDK
    #lst = [10703.0, 10704.0, 182305696.0, 182305712.0, 312782560.0, 312782592.0, 312782624.0, 312782656.0]
    #[args.remove(i) for i in lst]
    
    #setup
    nann = np.zeros(ann.shape).astype(bitdepth)
    cmap = np.linspace(0,65000, num=len(args)+1).astype(bitdepth)
    
    #populate
    for i in range(len(args)):
        nann[np.where(ann==args[i])] = cmap[i+1]
        if verbose and i%10 == 0: print ('{} of {}'.format(i, len(args)))
            
    
    return nann, annotation_value_to_structure(allen_id_table, args)

    
def consolidate_parents_structures(allen_id_table, ann, namelist, verbose=False):
    '''Function that generates evenly spaced pixels values based on annotation parents
    
    Removes 0 from list
    
    Inputs:
        allen_id_table=path to excel file generated from scripts above
        ann = allen annoation file
        namelist=list of structues names, typically parent structures*********************
        
    Returns:
        -----------
        nann = new array of bitdepth
        list of value+name combinations
    '''
    if type(ann) == str: ann = sitk.GetArrayFromImage(sitk.ReadImage(ann))

    df = pd.read_excel(allen_id_table)

    #remove duplicates and null and root
    namelist = list(set(namelist))
    namelist = [xx for xx in namelist if xx != 'null' and xx != 'root']
    
    #make structures to find parents
    from tools.analysis.network_analysis import make_structure_objects
    structures = make_structure_objects(allen_id_table)

    #setup
    nann = np.zeros(ann.shape).astype('uint8')
    cmap = [xx for xx in np.linspace(0,255, num=len(namelist))]
    
    #populate
    for i in range(len(namelist)):
        try:
            nm=namelist[i]
            s = [xx for xx in structures if xx.name==nm][0]
            if verbose: print ('{}, {} of {}, value {}'.format(nm, i, len(namelist)-1, cmap[i]))
            nann[np.where(ann==int(s.idnum))] = cmap[i]
            for ii in s.progeny: 
                if ii[3] != 'null': nann[np.where(ann==int(ii[3]))] = cmap[i]
        except Exception as e:
            print(nm, e)
    #sitk.Show(sitk.GetImageFromArray(nann))
            
    return nann, zip(cmap[:], namelist)


def consolidate_parents_structures_OLD(allen_id_table, ann, namelist, verbose=False):
    '''Function that generates evenly spaced pixels values based on annotation parents
    
    Removes 0 from list
    
    Inputs:
        allen_id_table=path to excel file generated from scripts above
        ann = allen annoation file
        namelist=list of structues names, typically parent structures*********************
        
    Returns:
        -----------
        nann = new array of bitdepth
        list of value+name combinations
    '''
    if type(ann) == str: ann = sitk.GetArrayFromImage(sitk.ReadImage(ann))

    df = pd.read_excel(allen_id_table)

    #remove duplicates and null and root
    namelist = list(set(namelist))
    namelist = [xx for xx in namelist if xx != 'null' and xx != 'root']
    
    #make structures to find parents
    from tools.analysis.network_analysis import make_structure_objects
    structures = make_structure_objects(allen_id_table)

    #setup
    nann = np.zeros(ann.shape).astype('uint8')
    cmap = [int(xx) for xx in np.linspace(0,255, num=len(namelist)+1)]
    
    #populate
    for i in range(len(namelist)):
        try:
            nm=namelist[i]
            s = [xx for xx in structures if xx.name==nm][0]
            if verbose: print ('{}, {} of {}, value {}'.format(nm, i, len(namelist)-1, cmap[i+1]))
            nann[np.where(ann==int(s.idnum))] = cmap[i+1]
            for ii in s.progeny: 
                if ii[3] != 'null': nann[np.where(ann==int(ii[3]))] = cmap[i+1]
        except Exception as e:
            print(nm, e)
    #sitk.Show(sitk.GetImageFromArray(nann))
            
    return nann, zip(cmap[1:], namelist)

if __name__ == '__main__':    
    
    if False: #do once
        #20180913 - remake with new annotations
        #http://api.brain-map.org/api/v2/structure_graph_download/1.json
        from tools.registration.allen_structure_json_to_pandas import *
        pth = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table2.json'
        pth = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.json'
        save = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx'
        allen_structure_json_to_pandas(pth, prune=True, ann_pth = None, save=save)
        #need to rename columns - since it's backwards
        df = pd.read_excel('/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')
        df = df.rename(columns={'parent_acronym':'parent_name', 'parent_name': 'parent_acronym'})
        #fix parent acronym and names
        for i,row in df.iterrows():
            if i!=0:df.loc[i, ['parent_name','parent_acronym']] = df[df.id==int(row['parent_structure_id'])][['name', 'acronym']].values[0]
        df.to_excel('/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')
        
        #add in voxel counts
        from tools.analysis.network_analysis import make_structure_objects
        ann_pth = '/jukebox/wang/pisano/Python/allenatlas/annotation_template_25_sagittal.tif'
        df_pth = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx'
        structures = make_structure_objects(df_pth, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
        ann = tifffile.imread(ann_pth)
        df = pd.read_excel('/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')
        df['voxels_in_structure'] = 0.0
        for s in structures:
            print(s.name)
            df.loc[df.name==s.name, 'voxels_in_structure']=len(np.where(ann==s.idnum)[0])     
        df.to_excel('/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table_w_voxelcounts.xlsx')
        
    
    df=pd.read_excel('/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')    
    ###search for structure using ID not atlas ID:
    df[df.name.str.contains('Caudoputamen')]
    ###or
    df[df.name=='Caudoputamen']
    ###or
    df[df.acronym=='VTA']
    
    ####scripts to isolate structures and overlay them
    #####
    #####NOTE ANNOTATION FILES HAVE BEEN UPDATED MAKE SURE USING CURRENT ANN/ATLAS FILES
    #####
    from tools.registration.allen_structure_json_to_pandas import *
    pth='/jukebox/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
    AtlasFile='/jukebox/wang/pisano/Python/allenatlas/average_template_25_sagittal.tif'
    svlc='/jukebox/wang/pisano/tracing_output/analysis/outlines'
    #TP atlas
    pth = '/jukebox/wang/pisano/Python/atlas/annotation_sagittal_atlas_20um_iso.tif'
    svlc = '/jukebox/wang/seagravesk/lightsheet/registration_tools'
    AtlasFile = '/jukebox/wang/pisano/Python/atlas/sagittal_atlas_20um_iso.tif'
    
    ###ACC    
    lst=[31, 572, 1053, 739, 179, 227, 39, 935, 211, 1015, 919,927,48,588,296,772,810,819]
    nm_to_sv='ACC'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    
    #Amygdala
    lst = df[df.name.str.contains('amygdalar')].id.tolist()
    nm_to_sv='Amygdala'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #Central Amyg
    lst = df[df.name.str.contains('Central amygdalar')].id.tolist()
    nm_to_sv='Central_Amygdala'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)    
     
    ###Lob6    
    lst=[936,10725,10724,10723]
    nm_to_sv='Lob6'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #sitk.Show(sitk.GetImageFromArray(im))
    ###lob7:
    lst=[944,10728,10727,10726]
    nm_to_sv='Lob7'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##ansiform lobule: 1017
    ###CrI,II turns out they don't actually have them separated
    lst=[1017, 1056, 10677,10676,10675, 1064,10680,10679,10678] 
    nm_to_sv='CrI,II'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##MDT
    lst = [617, 1077, 366, 362, 59, 636, 626]
    nm_to_sv='Medial group of the dorsal thalamus'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##VTA
    lst = [749]
    nm_to_sv='Ventral Tegmental Area'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##SNc
    lst = [381, 616, 374]
    nm_to_sv='Substantia Nigra'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##CrI
    lst = [1056]
    nm_to_sv='Crus I'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    ##PFC
    lst = [972, 171, 195, 304, 363, 84, 132]
    nm_to_sv='PFC'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #vestibular nucs
    lst = [640, 209, 202, 225, 217]
    nm_to_sv='Vestibular Nuclei'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #Pedunculopontine nucleus 
    lst = [1052]
    nm_to_sv='Pedunculopontine nucleus'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #Laterodorsal tegmental nucleus
    lst = [162]
    nm_to_sv='Laterodorsal tegmental nucleus'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #Pontine Central Gray
    lst = [892]
    nm_to_sv='Pontine Central Gray'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    #Pontine Reticular Nucs
    lst = [1093, 552, 146]
    nm_to_sv='Pontine Reticular Nucs'
    isolate_and_overlay(pth, AtlasFile, svlc, nm_to_sv, *lst)
    
if __name__=='__main__':
    allen_id_table = '/jukebox/wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx'
    annotation_value_to_structure(allen_id_table, [235, 345])
    
    #make annotation of stretched colors given a list
    nann, lst = isolate_structures_return_list(allen_id_table, ann, [1.0, 2.0, 4.0, 6.0])
    #nann, lst = isolate_structures_return_list(allen_id_table, ann,  list(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(ann))[100:175].ravel().astype('float64'))))

    #the args given will return zero :/
    lst = annotation_location_to_structure(allen_id_table, args=[(0, 77, 303), (0, 77, 304), (0, 78, 302)])
    
    #find area
    from tools.analysis.network_analysis import make_structure_objects
    ann_pth = '/jukebox/wang/pisano/Python/atlas/annotation_sagittal_atlas_20um_iso.tif'
    df_pth = '/jukebox/wang/pisano/Python/lightsheet/supp_files/ls_id_table_w_voxelcounts.xlsx'
    df = pd.read_excel(df_pth)
    structures = make_structure_objects(df_pth, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    thal_ids=[xx.idnum for xx in structures if 'Thalamus' in xx.progenitor_chain]
