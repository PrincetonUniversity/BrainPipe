# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:37:06 2016

@author: wanglab
"""
from tools.utils.io import listdirfull
from tools.analysis.network_analysis import make_structure_objects, variable_count
from tools.utils import find_all_brains
import numpy as np, os, cPickle as pickle, multiprocessing as mp, pandas as pd, sys, seaborn as sns
import matplotlib.pyplot as plt

def pth_update(inn):
    '''dummy function since we dont' need it and too lazy to fix'''
    return inn

def dendrogram_structures_vs_cerebellum(cores, pths, svlc, structure_of_interest, list_or_levels=False, boolean_for_cb = True, title=False, nametype = 'name', ann_pth=False, metric = False, method = False, cb_structures = False, analysismethod='division', return_arrays_only = False):
    '''Function to make seaborn dendrograms. y axis is progeny of structure_of_interest, x axis is added counts for cerebellar areas
    
    Inputs
    ---------------
    cores = number of cores to parallelize
    pths = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = output folder
    structure_of_interest (SOI)= exact NAME of ABA structure to use, outputs all progeny
    nametype = 'name' for full names, 'acronym' for acronyms
    title = plot title
    list_or_levels = False: do nothing; 
                     Integer: the level of structures to list; 
                     list = explicit list of structure NAMES to use
    boolean_for_cb = 
                True: 'biniarizes cerebellar' matrix such that if there is any "cell count" then it adds to the SOI total
                False: matrix multiplication between cerebellar count region and SOI count
    cb_structures =
                False: uses all cb structures: 'Ansiform lobule','Central lobule','Copula pyramidis','Culmen','Declive (VI)','Dentate nucleus','Fastigial nucleus','Flocculus','Folium-tuber vermis (VII)','Interposed nucleus','Lingula (I)','Nodulus (X)','Paraflocculus','Paramedian lobule','Pyramus (VIII)','Simple lobule','Uvula (IX)'
                lst of cb structures to use: i.e. ['Ansiform lobule','Declive (VI)','Dentate nucleus','Fastigial nucleus', 'Interposed nucleus', 'Paramedian lobule','Pyramus (VIII)']
                note: if lst type must reflect nametype, i.e. w/ nametype=name then list must be names, same for acronym
    analysismethod=
                'division' : need to check if mathematically sound - currently dot product of m[SOI, brain] * m[brain, cerebellar region]^-1
                'dotproduct': older version, 
    return_arrays_only = True returns: brainnames, name_or_acronym, data_array, cb_data_array, cb_name_or_acronym 
    ###NOTE: 
    https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html
    things to chance for clustering:
    method = 'average', 'complete' 'average', 'centroid', etc...:see scipy.cluster.hierarchy.linkage
    metric = 'correlation' ; http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    '''    
    
    #https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html
    makedir(svlc)
    if not ann_pth: ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015_reflectedhorizontally.nrrd'
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct


    ###structures of interest###
    sys.stdout.write('\nStarting parallelization for {}: ~2mins...'.format(structure_of_interest)) 
    p=mp.Pool(cores)
    iterlst=[]; [iterlst.append((ann_pth, structure_of_interest, name, pth, nametype, list_or_levels)) for name, pth in pths.iteritems()]
    nlst = p.starmap(make_acronym_count_dictionary, iterlst); nlst.sort()
    p.terminate()
    sys.stdout.write('done') 

    #unpack after parallelization
    name_or_acronym = [xx for xx in nlst[0].values()[0].iterkeys()]; name_or_acronym.sort()
    brainnames = [xx.keys()[0] for xx in nlst]
    
    #initialize np array
    data_array = np.zeros((len(brainnames),len(name_or_acronym)))
    
    #fill array    
    for yindx in range(len(name_or_acronym)):
            for xindx in range(len(nlst)):
                        data_array[xindx, yindx] = nlst[xindx].values()[0][name_or_acronym[yindx]]

    
    ###cb structures###    
    ##make for cerebellar areas --- this will likely change
    sys.stdout.write('\nStarting parallelization for Cerebellum: ~2mins...'.format(structure_of_interest)) 
    p=mp.Pool(cores); cb_levels = 3
    iterlst=[]; [iterlst.append((ann_pth, 'Cerebellum', name, pth, nametype, cb_levels)) for name, pth in pths.iteritems()]
    cblst = p.starmap(make_acronym_count_dictionary, iterlst); cblst.sort()
    p.terminate()
    sys.stdout.write('done') 
    
    #unpack after parallelization
    cb_name_or_acronym = [xx for xx in cblst[0].values()[0].iterkeys()]; cb_name_or_acronym.sort()
    cb_brainnames = [xx.keys()[0] for xx in cblst]
    
    #initialize np array
    cb_data_array = np.zeros((len(cb_brainnames),len(cb_name_or_acronym)))

    for yindx in range(len(cb_name_or_acronym)):
            for xindx in range(len(cblst)):
                cb_data_array[xindx, yindx] = cblst[xindx].values()[0][cb_name_or_acronym[yindx]]

    if return_arrays_only: return brainnames, name_or_acronym, data_array, cb_data_array, cb_name_or_acronym 
    '''
    ##remove brains that have zero for all values (column):
    cols_to_rm = np.all(data_array == 0, axis = 1)
    rmlst = []
    for x in range(len(cols_to_rm)):
        if cols_to_rm[x]: rmlst.append(x)
    data_array = np.delete(data_array, rmlst, axis=0)
    cb_data_array = np.delete(cb_data_array, rmlst, axis=0) 
    '''
    ###COMPUTATIONAL PART, NEED TO DETERMINE BEST WAY TO DO THIS
    if analysismethod=='dotproduct':
        #combine two dataarrays: matrix multiplication of (SOI, brains) w (brains, cerebellar region)
        #this might not be the optimal way to analyze
        if boolean_for_cb: n_data_array = np.dot(data_array.transpose(), np.asarray(cb_data_array, dtype=bool)) 
        if not boolean_for_cb: n_data_array = np.dot(data_array.transpose(), cb_data_array)
    elif analysismethod=='division':
        #combine two dataarrays-- per brain: divide regional cell counts by postive cerebellar pixels
        #(SOI, brains) w (brains, cerebellar region)
        #FIXME: make sure this is mathematically sound
        if boolean_for_cb: n_data_array = np.dot(data_array.transpose(), np.linalg.inv(np.asarray(cb_data_array, dtype=bool)))
        if not boolean_for_cb: n_data_array = np.dot(data_array.transpose(), np.linalg.inv(cb_data_array))
        
        
    #make dataframe; can remove .transpose()
    df = pd.DataFrame(n_data_array.transpose(), index=cb_name_or_acronym, columns=name_or_acronym).transpose()
    
    #remove cb structures not of interest
    if cb_structures: 
        for xx in set(df.columns).difference(cb_structures):
            exec('del df["{}"]'.format(xx))
        
    #save dataframe
    #df.to_csv(os.path.join(svlc, 'dataframe_for_{}_vs_cerbellum_using_{}s'.format(structure_of_interest, nametype)))
    df.to_excel(os.path.join(svlc, 'dataframe_for_{}_vs_cerbellum_using_{}s.xls'.format(structure_of_interest, nametype)))
    
    #remove columns that are all zeros (column w all zeros breaks clustermap function):
    tmp = df.loc[:, (df !=0).any(axis=0)]
    sys.stdout.write('\n\n***REMOVING {}***\n   as no cells were detected (i.e. all structures of interest had zero counts) w/o this may break clustermap function\n\n'.format(list(df.columns-tmp.columns)))
    df = tmp.transpose()
    tmp = df.loc[:, (df !=0).any(axis=0)]
    sys.stdout.write('\n\n***REMOVING {}***\n   as no cells were detected (i.e. all brains had zero counts for structure of interest) w/o this may break clustermap function\n\n'.format(list(df.columns-tmp.columns)))
    df = tmp.tranpose()
    
    #set param
    cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    #cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    
    if not metric: metric = 'correlation'
    if not method: method = 'weighted'
    #normal weighted; to see nonweighted use = 'complete'
    g = sns.clustermap(df, cmap=cmap, method = method, metric = metric)
    #g.fig.text(0.035, .5,'{}'.format(structure_of_interest), fontsize=20, rotation=90)
    #g = sns.clustermap(df, row_cluster=False, cmap=cmap, method = 'weighted') # for non clustered rows    
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.2)
    if title:
        if type(list_or_levels) == list: g.fig.suptitle('{}: Weighted ClusterMap, using {} metric and {} clustering method'.format(title, metric, method))
        if not type(list_or_levels) == list: g.fig.suptitle('{}: Weighted ClusterMap of {}, using {} metric and {} clustering method'.format(title, structure_of_interest, metric, method))
    else:
        g.fig.suptitle('Weighted ClusterMap of {}, using {} metric and {} clustering method'.format(structure_of_interest, metric, method))
    g.savefig(os.path.join(svlc, 'dendrogramheatmap_for_{}_vs_cerebellum_using_{}_and_{}_clustering_method'.format(structure_of_interest, metric, method)), dpi=450)    

    #to get linkage matrix:
    #col_linkage = g.dendrogram_col.linkage # linkage matrix for columns
    #row_linkage = g.dendrogram_row.linkage # linkage matrix for rows

    #standarized
    try:    
        g = sns.clustermap(df, cmap=cmap, method = method, standard_scale=1, metric = metric) #
        #g.fig.text(0.035, .5,'{}'.format(structure_of_interest), fontsize=20, rotation=90)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.2)
        if title:
            if type(list_or_levels) == list: g.fig.suptitle('{}: Standardized ClusterMap, using {} metric and {} clustering method'.format(title, metric, method))
            if not type(list_or_levels) == list: g.fig.suptitle('{}: Standardized ClusterMap of {}, using {} metric and {} clustering method'.format(title, structure_of_interest, metric, method))
        else:
            g.fig.suptitle('Standardized ClusterMap of {}, using {} metric and {} clustering method'.format(structure_of_interest, metric))
        g.savefig(os.path.join(svlc, 'dendrogramheatmap_standardized_for_{}_vs_cerebellum_using_{}_and_{}_clustering_method'.format(structure_of_interest, metric, method)), dpi=450)    
    except:
        print ('Standarization failed')

    '''
    #normalizing data across rows
    g = sns.clustermap(df, cmap=cmap, method = 'weighted', z_score=0)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.35)
    g.fig.suptitle('{}: Zscored ClusterMap of {}_vs_cerebellum'.format(title, structure_of_interest)) if title else g.fig.suptitle('Zscored ClusterMap of {}'.format(structure_of_interest))
    g.savefig(os.path.join(svlc, 'dendrogramheatmap_zscored_for_{}'.format(structure_of_interest)))
    '''
    
    return 
    

def dendrogram_structures_by_brain(cores, pths, svlc, structure_of_interest, list_or_levels = False, title=False, nametype = 'name', ann_pth = False, metric = False, method=False, nmcolorlst = False):
    '''Function to make seaborn dendrograms. y axis is progeny of structure_of_interest, x axis is brains
    
    Inputs
    ---------------
    cores = number of cores to parallelize
    pths = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = output folder
    structure_of_interest = exact NAME of ABA structure to use, outputs all progeny
    nametype = 'name' for full names, 'acronym' for acronyms
    title = title for figure
    list_or_levels = False: do nothing; integer: the level of structures to list; list = explicit list of structure NAMES to use
    nmcolorlst = zip(brainnames, colorlst); list of [brainname, color]
    
    ###NOTE: 
    https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html
    things to chance for clustering:
    method = 'average', 'complete' 'average', 'centroid', etc...:see scipy.cluster.hierarchy.linkage
    metric = 'correlation' ; http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    '''    
    
    #https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html
    makedir(svlc)
    if not ann_pth: ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015_reflectedhorizontally.nrrd'
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct
    
    ##make lists
    sys.stdout.write('\nStarting parallelization for {}: ~2mins...'.format(structure_of_interest)) 
    p=mp.Pool(cores)
    iterlst=[]; [iterlst.append((ann_pth, structure_of_interest, name, pth, nametype, list_or_levels)) for name, pth in pths.iteritems()]
    nlst = p.starmap(make_acronym_count_dictionary, iterlst); nlst.sort()
    p.terminate()
    sys.stdout.write('done') 
    
    #unpack after parallelization
    name_or_acronym = [xx for xx in nlst[0].values()[0].iterkeys()]; name_or_acronym.sort()
    brainnames = [xx.keys()[0] for xx in nlst]
    
    #initialize np array
    data_array = np.zeros((len(brainnames),len(name_or_acronym)))
    
    #fill array    
    brainnames=range(len(nlst))
    for yindx in range(len(name_or_acronym)):
            for xindx in range(len(nlst)):
                brainnames[xindx]=nlst[xindx].keys()[0]
                for acr, count in nlst[xindx].values()[0].iteritems():
                    if acr == name_or_acronym[yindx]:
                        data_array[xindx, yindx] = count
    
    #make dataframe; can remove .transpose()
    df = pd.DataFrame(data_array, index=brainnames, columns=name_or_acronym).transpose()
    #save dataframe
    df.to_excel(os.path.join(svlc, 'dataframe_for_{}_using_{}s.xls'.format(structure_of_interest, nametype)))
    
    #remove columns that are all zeros (column w all zeros breaks clustermap function):
    aa = df.loc[:, (df !=0).any(axis=0)]
    sys.stdout.write('\n\n***REMOVING {}***\n   as no cells were detected (i.e. all structures of interest had zero counts) w/o this may break clustermap function\n\n'.format(list(df.columns-aa.columns)))
    df = aa
    
    #set param
    cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    #cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    if not metric: metric = 'correlation'
    if not method: method = 'weighted'
    #normal weighted; to see nonweighted use = 'complete'
    if not nmcolorlst: g = sns.clustermap(df, cmap=cmap, method = method, metric = metric)
    if nmcolorlst: 
        colorlst = [xx[1] for yy in df.columns for xx in nmcolorlst if yy==xx[0]] #done to map colors appropriately
        g = sns.clustermap(df, cmap=cmap, method = method, metric = metric, col_colors=colorlst)
    #g.fig.text(0.035, .5,'{}'.format(structure_of_interest), fontsize=20, rotation=90)
    #g = sns.clustermap(df, row_cluster=False, cmap=cmap, method = 'weighted') # for non clustered rows    
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.15)
    if title:
        if type(list_or_levels) == list: g.fig.suptitle('{}: Weighted ClusterMap, using {} metric and {} clustering method'.format(title, metric, method))
        if not type(list_or_levels) == list: g.fig.suptitle('{}: Weighted ClusterMap of {}, using {} metric and {} clustering method'.format(title, structure_of_interest, metric, method))
    else:
        g.fig.suptitle('Weighted ClusterMap of {}, using {} metric and {} clustering method'.format(structure_of_interest, metric, method))
    g.savefig(os.path.join(svlc, 'dendrogramheatmap_for_{}_using_{}_and_{}_clustering method'.format(structure_of_interest, metric, method)), dpi=450)    
    
    #to get linkage matrix:
    #col_linkage = g.dendrogram_col.linkage # linkage matrix for columns
    #row_linkage = g.dendrogram_row.linkage # linkage matrix for rows

    #standarized
    try:
        if not nmcolorlst: g = sns.clustermap(df, cmap=cmap, method = method, standard_scale=1, metric = metric)        
        if nmcolorlst: g = sns.clustermap(df, cmap=cmap, method = method, standard_scale=1, metric = metric, col_colors=colorlst)
        #g.fig.text(0.035, .5,'{}'.format(structure_of_interest), fontsize=20, rotation=90)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.15)
        if title:
            if type(list_or_levels) == list: g.fig.suptitle('{}: Standardized ClusterMap, using {} metric and {} clustering method'.format(title, metric, method))
            if not type(list_or_levels) == list: g.fig.suptitle('{}: Standardized ClusterMap of {}, using {} metric and {} clustering method'.format(title, structure_of_interest, metric, method))
        else:
            g.fig.suptitle('Standardized ClusterMap of {}, using {} metric and {} clustering method'.format(structure_of_interest, metric,method))
        g.savefig(os.path.join(svlc, 'dendrogramheatmap_standardized_for_{}_using_{}_and_{}_clustering method'.format(structure_of_interest, metric, method)), dpi=450)
    except:
        print ('Dendrogram Standardization failed')

    '''
    #normalizing data across rows
    g = sns.clustermap(df, cmap=cmap, method = 'weighted', z_score=0)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.35)
    g.fig.suptitle('{}: Zscored ClusterMap of {}'.format(title, structure_of_interest)) if title else g.fig.suptitle('Zscored ClusterMap of {}'.format(structure_of_interest))
    g.savefig(os.path.join(svlc, 'dendrogramheatmap_zscored_for_{}'.format(structure_of_interest)))
    '''
    
    return

#%%

def make_acronym_count_dictionary(ann_pth, structure_of_interest, name, pth, nametype, list_or_levels):
    
    '''Helper function for parallelization of structures counts
    
    Outputs
    ------------------
    dictionary = {brainname: {acronym : cell count}}
    '''       
    if directorydeterminer() == '/home/wanglab/': param_dict = 'param_dict_local.p'
    if not directorydeterminer() == '/home/wanglab/': param_dict = 'param_dict.p'    
    kwargs={}  
    try:        
        with open(os.path.join(pth, param_dict), 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()
    except IOError:
        with open(os.path.join(pth, 'param_dict.p'), 'rb') as pckl:
            kwargs.update(pickle.load(pckl))
            pckl.close()
            
    if directorydeterminer() == '/home/wanglab/': kwargs.update(pth_update(kwargs))
    
    #load    
    try:        
        excelfl = [xx for xx in listdirfull(pth) if 'cellch' in xx and '.xlsx' in xx][0]
    except IndexError:
        excelfl = [xx for xx in listdirfull(pth) if '.xlsx' in xx][0]
    
    #generate objects:
    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
      
    #if list_or_levels != False:
    if type(list_or_levels) == int:
        structure_n_cell_count = variable_count(structures, structure_of_interest, list_or_levels, nametype=nametype); structure_n_cell_count = [list(xx) for xx in structure_n_cell_count]
    
    elif type(list_or_levels) == list:
        if nametype == 'name': structure_n_cell_count = [[xx.name , xx.cellcount_progeny] for xx in structures for yy in list_or_levels if xx.name == yy]
        if nametype == 'acronym': structure_n_cell_count = [[xx.acronym , xx.cellcount_progeny] for xx in structures for yy in list_or_levels if xx.name == yy]
    
    else:
        #structure of interest        
        try:
            soi = [xx for xx in structures if structure_of_interest in xx.name][0]
        except IndexError:
            print('Structure of Interest not found, formatting must be EXACT')
        #unpack labels(structures), and cell counts    
        if nametype == 'name': structure_n_cell_count = [[xx.name, xx.cellcount_progeny] for prog in soi.progeny for xx in structures if xx.acronym == prog[2]]
        if nametype == 'acronym': structure_n_cell_count = [[xx.acronym, xx.cellcount_progeny] for prog in soi.progeny for xx in structures if xx.acronym == prog[2]]
    
    #if you want just cell counts        
    #acronyms, counts = zip(*[[xx.acronym, xx.cellcount] for prog in soi.progeny for xx in structures if xx.acronym == prog[2]])
    
    #package into dictionary==>{brainname: dict[acronymname]=cellcount}
    dct = {name : {}}
    for structure_count in structure_n_cell_count:
        dct[name][structure_count[0]] = structure_count[1]    

    return dct
     
    
if __name__ == '__main__':
    #trying to make linkage of heirachy from aba
    from  scipy.cluster.hierarchy import linkage
    
    a = linkage(df.transpose())
    b = linkage(df)
    
    
    df=pd.read_excel('/home/wanglab/temp_wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx')    
    id_parentid = [[df.iloc[x][1], df.iloc[x][2], df.iloc[x][4]] for x in range(len(df))];
    
    array = np.zeros((2, len(id_parentid)))
    
    df = pd.DataFrame.from_csv('/home/wanglab/Desktop/tmp/dataframe_for_Thalamus_using_names')
    
    
    ##order the linkage array to be in order of the structures
    for xx in range(len(df)):
        for ii in range(len(id_parentid)):
            name,a,b = id_parentid[ii]        
            if df.iloc[xx][0] == name:
                if b == 'null': b = 0
                if a == 'null': a = 0
                array[:,xx] = a,b
    #linkage
    row_linkarr = linkage(array)
    
    #normal weighted; to see nonweighted use = 'complete'
    g = sns.clustermap(df, cmap=cmap, method = 'weighted', figsize = (10,10))
    #g = sns.clustermap(df, row_cluster=False, cmap=cmap, method = 'weighted') # for non clustered rows    
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.35)
    g.savefig(os.path.join(svlc, 'dendrogramheatmap_for_thalamus_attempt_atlinkage'))
    
    #testing
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    #ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015_reflectedhorizontally.nrrd'
    lob6 = [xx for xx in find_all_brains(tracing_output_fld) if 'sd_hsv_lob6' in xx]; lob6.sort()
    names = ['Lob6 250U 1050R','Lob6 250U 750R', 'Lob6 500U 900R', 'Lob6 500D ML0', 'Lob6 250U 150R', 'Lob6 500D 300R', 'Lob6 250U 450R']
    lob6dct = dict(zip(names, lob6))

    #tmp
    del lob6dct['Lob6 500D ML0']
    
    #
    svlc = '/home/wanglab/wang/pisano/tracing_output/analysis/dendrograms/sd_lob6'
    cores = 7
    list_or_levels = 3 #can ==False
    title = 'SD HSV Lob6'
    pths = lob6dct
    #per brain
    [dendrogram_structures_by_brain(cores, pths, svlc, structure_of_interest, list_or_levels=2, title = title) for structure_of_interest in ['Thalamus', 'Isocortex']]
    [dendrogram_structures_by_brain(cores, pths, svlc, structure_of_interest, list_or_levels = list_or_levels, title = title) for structure_of_interest in ['Hypothalamus', 'Cerebellum', 'Cerebral nuclei', 'Hippocampal region', 'Hindbrain', 'Interbrain', 'Brain stem', 'Midbrain', 'Medulla', 'Pons', 'Cortical subplate']]
    
    #vs cerebellar structure (you will need to look at how to determine cb counts and boolean_for_cb)
    dendrogram_structures_vs_cerebellum(cores, pths, svlc, 'Thalamus', list_or_levels, boolean_for_cb = True, title = title, nametype = 'name')

    #using lists:
    list_or_levels = ['Anterior cingulate area','Substantia nigra, reticular part', 'Visual areas', 'Somatosensory areas','Hippocampal region','Frontal pole, cerebral cortex','Striatum ventral region','Infralimbic area','Pallidum, dorsal region', 'Prelimbic area','Midbrain raphe nuclei','Retrosplenial area','Somatomotor areas','Red nucleus','Lateral septal complex','Thalamus, polymodal association cortex related','Temporal association areas','Ectorhinal area','Gustatory areas','Striatum-like amygdalar nuclei','Orbital area','Visceral area','Perirhinal area','Posterior parietal association areas','Ventral tegmental area','Claustrum','Pallidum','Pedunculopontine nucleus','Periaqueductal gray','Substantia nigra, compact part','Agranular insular area','Auditory areas','Thalamus, sensory-motor cortex related', 'Hypothalamus','Nucleus of Darkschewitsch']
    [dendrogram_structures_by_brain(cores, pths, svlc, structure_of_interest, title = 'SD HSV Lob6 Selected Structures', list_or_levels=list_or_levels) for structure_of_interest in ['Selected Structures']]    
    dendrogram_structures_vs_cerebellum(cores, pths, svlc, 'Selected Structures', list_or_levels, boolean_for_cb = True, title = 'SD HSV Lob6 Selected Structures', nametype = 'name')
    
    #different metrics   
    svlc = '/home/wanglab/wang/pisano/tracing_output/analysis/dendrograms/metrics'
    [dendrogram_structures_by_brain(cores, pths, svlc, 'Thalamus', list_or_levels = 2, title = title, metric = metric) for metric in ['euclidean', 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'braycurtis']]
    [dendrogram_structures_by_brain(cores, pths, svlc, 'Thalamus', list_or_levels = 2, title = title, metric = metric) for metric in ['matching', 'dice', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'wminkowski']]
    #[dendrogram_structures_by_brain(cores, pths, svlc, 'Thalamus', list_or_levels = 2, title = title, metric = metric) for metric in ['euclidean', 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'braycurtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'wminkowski']]
    
    #colorlst:
    colorlst = ['darksalmon', 'lightgreen', 'skyblue', 'b', 'r', 'g', 'm', 'y', 'c', 'darkturquoise', 'lime', 'firebrick', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']
    nmcolorlst = zip(pths.keys(), colorlst)
    dendrogram_structures_by_brain(cores, pths, svlc, 'Thalamus', list_or_levels = 2, title = title, metric = 'cityblock', nmcolorlst = nmcolorlst) 
    