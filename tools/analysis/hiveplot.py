# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:57:53 2016

@author: wanglab
"""
from __future__ import division
from tools.utils.io import makedir, listdirfull, makedir, removedir, chunkit, writer, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer
from tools.analysis.network_analysis import make_structure_objects, make_network, find_structures_of_given_level
import pandas as pd
from hiveplot import HivePlot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def pth_update(inn):
    '''dummy function since we dont' need it and too lazy to fix'''
    return inn
#%%


if __name__ == '__main__':    
    
    
    ##TO FIND NEW STRUCTURE IDs    
    excelfl = '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/ch00_l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_stuctures_table.xlsx'
    #structure_id= 549 #Thalamus's id NOT atlas_id
    ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
    
    #load    
    df = pd.read_excel(excelfl)

    #generate objects:
    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth, verbose=True)
    
    #optional to find progeny of given level:
    cutofflevel = 3
    substructures = find_structures_of_given_level('Basic cell groups and regions', cutofflevel, structures)    
    
    #find structure numbers:
    strct = [xx for xx in structures if xx.name == 'Thalamus'][0]
    ids = [xx[3] for xx in strct.progeny]; ids.append(strct.idnum)


#%%

#testing
inputexcel = '/home/wanglab/wang/pisano/tracing_output/prv/jl_20160621_bl6_md45/jl_20160621_bl6_md45_488w_555_z3um_1hfds_005na_150msec_injch_stuctures_table.xlsx'
inputexcel = '/home/wanglab/wang/pisano/tracing_output/bl6_crI/db_20160616_cri_53hr/db_20160616_cri_53hr_488w_561_200msec_z3um_1hfds_injch_stuctures_table.xlsx'
outputexcel = '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/ch00_l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_stuctures_table.xlsx'
    
    
#%%
def make_hiveplot(inputexcel, outputexcel, node1='Cerebellum', node2='Thalamus', node3=False, colorlst=['black', 'red', 'white'], title=False, annpath = False, threshold = 0.25):
    '''leveraging off of: https://github.com/ericmjl/hiveplot
    
    Inputs:
    --------------------------
    
    inputexcel = excel file of injection site or starting point
    outputexcel = excel file of cell counts or ending points
    node1-3: ABA Name of structures. node 1 for inputexcel, node2-3 for outputexcel
    threshold = value above to keep. This is a 'score' of connectivity: ((injectionpixels/maxinjectionarea pixels) + (cellcount/maxcellcount)) /  2
    '''    

    #################TESTING:
    node1='Cerebellum'
    node2='Thalamus'
    node3='Cerebrum'
    ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'    
    colorlst=['black', 'red', 'blue']
    title = 'Sample HivePlot'
    #
    if not ann_pth: ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'    
    
    #generate objects:
    innstructures = make_structure_objects(inputexcel, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    outstructures = make_structure_objects(outputexcel, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)

    #make networkx graphs    
    Ginn = make_network(innstructures, substructure_name = 'Basic cell groups and regions', graphstyle = 'twopi', showcellcounts = False, cutofflevel=10, font_size = 11, show = False)        
    Gout = make_network(outstructures, substructure_name = 'Basic cell groups and regions', graphstyle = 'twopi', showcellcounts = False, cutofflevel=10, font_size = 11, show = False)        
    

    #find structure objects for nodes of given input
    nn1 = [xx for xx in innstructures if xx.name == node1][0]
    nn2 = [xx for xx in outstructures if xx.name == node2][0] 
    if node3: nn3 = [xx for xx in outstructures if xx.name == node3][0]
    
    #pull out progeny of nodes; might need to make this level based
    nodes = dict()
    nodes['group1'] = [(n, d['structure_object'].cellcount_progeny) for n, d in Ginn.nodes(data=True) if d.values()[1].name in [str(xx[1]) for xx in nn1.progeny]] #
    nodes['group2'] = [(n, d['structure_object'].cellcount_progeny) for n, d in Gout.nodes(data=True) if d.values()[1].name in [str(xx[1]) for xx in nn2.progeny]] #
    if node3: nodes['group3'] = [(n, d['structure_object'].cellcount_progeny) for n, d in Gout.nodes(data=True) if d.values()[1].name in [str(xx[1]) for xx in nn3.progeny]] #
    
    #sort, might need to change sorting measure; currently by count
    for group, nodelist in nodes.items():
        nodelist.sort(key=lambda x: x[1], reverse = True) #, key=keyfunc())
        nodes[group] = [xx[0] for xx in nodelist]
    
    #pull out node ids
    #nodes[group] = [n for n, d in nodes[group]]
    

    #lst of [name, counts]:
    n1lst = [[i, [xx.cellcount for xx in innstructures if xx.name == i][0]] for i in nodes['group1']]
    n2lst = [[i, [xx.cellcount for xx in outstructures if xx.name == i][0]] for i in nodes['group2']]
    if node3: n3lst = [[i, [xx.cellcount for xx in outstructures if xx.name == i][0]] for i in nodes['group3']]
    
    #create normalization    
    n1max = max([xx[1] for xx in n1lst])
    n2max = max([xx[1] for xx in n2lst])
    n3max = max([xx[1] for xx in n3lst])
    
    #FIXME: not yet implemented a true sorting value...you should filter by third number = 0->1
    edges = dict()
    edges['group1'] = [(xx[0],yy[0], (((xx[1]/n1max)+(yy[1]/n2max))/2)) for xx in n1lst for yy in n2lst if xx[1] > 0 and yy[1] > 0 and (((xx[1]/n1max)+(yy[1]/n2max))/2) > threshold]
    if node3: edges['group2'] = [(xx[0],yy[0], (((xx[1]/n1max)+(yy[1]/n3max))/2)) for xx in n1lst for yy in n3lst if xx[1] > 0 and yy[1] > 0 and (((xx[1]/n1max)+(yy[1]/n3max))/2) > threshold]
    
    
    #set colors
    nodes_cmap = dict()
    nodes_cmap['group1'] = colorlst[0]
    nodes_cmap['group2'] = colorlst[1]
    if node3: nodes_cmap['group3'] = colorlst[2]
    
    edges_cmap = dict()
    edges_cmap['group1'] = (0, 0.5, 0.5)
    if node3: edges_cmap['group2'] = (0.2, 0.4, 0.9)
    
    
    #plot!
    f = plt.figure()
    ax = f.gca(); ax.set_axis_bgcolor('black')
    h = HivePlot(nodes, edges, nodes_cmap, edges_cmap, is_directed=True, fig = f, linewidth = 0.3, ax = ax)
    if title: f.suptitle(title, fontsize=14, fontweight='bold')
    legend = [mpatches.Patch(color = colorlst[0], label = node1), mpatches.Patch(color = colorlst[1], label = node2)]
    if node3: legend.append(mpatches.Patch(color = colorlst[2], label = node3))
    f.text(0.05,0.2,'{} --> {}'.format(node1, node2), color = edges_cmap['group1'], backgroundcolor = (0.6, 0.6 ,0.6), fontsize = 14)
    if node3: f.text(0.05,0.15,'{} --> {}'.format(node1, node3), color = edges_cmap['group2'], backgroundcolor = (0.6, 0.6 ,0.6), fontsize = 14)
    h.draw()

#%%

import networkx as nx
import numpy as np

from hiveplot import HivePlot



G = nx.read_gpickle('/home/wanglab/Desktop/test/hiveplot/tests/test_data/test_graph.pkl')


groups = [1,2,3]

nodes = dict()

for g in groups:
    nodes[g] = [n for n, d in G.nodes(data=True) if d['group'] == g]
    
nodes_cmap = dict(zip(groups, ['red', 'green', 'blue']))

edges = dict()
edges['group1'] = []
for u,v,d in G.edges(data=True):
    edges['group1'].append((u,v,d))


h = HivePlot(nodes, edges, nodes_cmap)
h.set_minor_angle(np.pi / 32)
h.draw()
    
    
