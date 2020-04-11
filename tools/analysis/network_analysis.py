# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:56:43 2016

@author: wanglab
"""

import networkx as nx
import matplotlib.pyplot as plt    
import pandas as pd    
#import pygraphviz as pg
from networkx.drawing.nx_agraph import graphviz_layout #from networkx.drawing.nx_agraph import write_dot()
from itertools import compress
from math import sqrt, ceil
import sys, collections
from tools.imageprocessing.preprocessing import listdirfull

class structure:
    """Class to represent a brain structure
    """
    kind="structure"    
    def __init__(self, idnum, excelfl, units="pixels", scale_factor=1):
        self.idnum=float(idnum)        #id of structure 
        self.idnum_int = int(idnum)    #id of structure as float, sometimes useful for comparisions
        self.excelfl=excelfl            #path to excelfl
        self.acronym=()                 #acronym of structure        
        self.name=()                    #allen name of strcutre
        self.cellcount=()               #determined count of actual structure
        self.cellcount_progeny=()       #determined count of progeny + structure
        self.parent=()                  #parent id, acronym, name
        self.children=[]                #children one level down (first sublevel)
        self.progeny=[]                 #all children, grandchildren, etc
        self.progeny_pixels=[idnum]     #all progeny pixel IDs and it"s own
        self.progenitor_chain=[]        #list moving up the heirarchy. i.e.: LGN->Thal->Interbrain
        self.volume=()                  #number of voxels, scale factor is not accounted for unless provided
        self.volume_progeny=()           #number of voxels, scale factor is not accounted for unless provided of progeny+structure
        self.units=units                #units that volume is displayed as
        self.scale_factor = scale_factor#unittopixel conversion. I.e. aba=25um/pixels == 25
        
    def add_name(self, nm): 
        self.name=nm
    def add_acronym(self, acronym): 
        self.acronym=acronym
    def add_cellcount(self, num): 
        self.cellcount=num
    def add_cellcount_progeny(self, num): 
        self.cellcount_progeny=num
    def add_parent(self, nm): 
        self.parent=nm
    def add_child(self, nm): 
        self.children.append(nm)
    def add_progeny(self, nm): 
        self.progeny.append(nm)
    def add_progeny_pixels(self, nm): 
        self.progeny_pixels.append(nm)
    def create_progenitor_chain(self, lst):
        self.progenitor_chain=lst

def find_progeny(struct, df):
    #find children (first sublevel)   
    #children = [x for x in df[df["parent_structure_id"] == str(struct.idnum)].itertuples()]
    children = [x for x in df[df["parent_structure_id"] == struct.idnum].itertuples()]
    
    #find all progeny
    allchildstructures=[]
    while len(children) > 0:
        child = children.pop()
        allchildstructures.append(child)
        #kiditems = df[df["parent_structure_id"] == str(child.id)]
        kiditems = df[df["parent_structure_id"] == child.id]
        for kid in kiditems.itertuples():
            allchildstructures.append(kid)            
            #if kid has children append them list to walk through it
            #if len(df[df["parent_structure_id"] == str(kid.id)]) != 0:
            if len(df[df["parent_structure_id"] == kid.id]) != 0:                
                children.append(kid)
    #remove duplicates
    allchildstructures = list(set(allchildstructures))
    #add progeny to structure_class
    [struct.add_progeny(xx) for xx in allchildstructures]
    #add_progeny_pixels to structure_class
    [struct.add_progeny_pixels(xx.id) for xx in allchildstructures] #<---11/1/17, this was atlas_id, changing to id
    #add progeny count
    struct.add_cellcount_progeny(struct.cellcount + sum([int(xx.cell_count) for xx in allchildstructures]))
    if "voxels_in_structure" in df.columns: struct.volume_progeny = struct.volume + sum([int(xx.voxels_in_structure) for xx in allchildstructures])
    
    return struct
    
def create_progenitor_chain(structures, df, verbose = False):

    new_structures=[]
    for struct in structures:
        if verbose: print(struct.name)
        if struct.name == "root" or struct.name == "Basic cell groups and regions":
            pass
        else:
            chain = []
            loop = True
            current_struct = struct
            while loop:
                #find parent
                parent = current_struct.parent[1]
                if parent == "nan" or parent == "null" or parent == "root": break
                else:#append
                    chain.append(parent)
                    current_struct = [xx for xx in structures if xx.name == parent][0]
            struct.create_progenitor_chain(chain)
            new_structures.append(struct)    

    return new_structures

def make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = False, ann_pth = None, verbose = False):

    #load
    df = pd.read_excel(excelfl)
    if not "cell_count" in df.columns: df["cell_count"] = 0
    #make structure objects for each df row
    structures = []
    
    for row in df.itertuples():
        #basic structure class features
        struct = structure(row.id , excelfl) #make object using ID and location of excelfl
        struct.add_name(str(row.name)) #add name        
        struct.add_acronym(str(row.acronym)) #add acronym
        struct.add_cellcount(row.cell_count) #add cell count
        struct.add_parent((row.parent_structure_id, str(row.parent_name), str(row.parent_acronym))) #parent id, acronym, name
        if "voxels_in_structure" in df.columns: struct.volume = row.voxels_in_structure
        
        #find children (first sublevel)   
        #children = [x for x in df[df["parent_structure_id"] == str(struct.idnum_int)].itertuples()]
        children = [x for x in df[df["parent_structure_id"] == struct.idnum_int].itertuples()]
        struct.add_child([xx for xx in children])
              
        #add structure to structures list
        structures.append(struct)
    #find progeny (all sublevels)
    structures = [find_progeny(struct, df) for struct in structures]
    
    #create progenitor_chain
    structures = create_progenitor_chain(structures, df)
    ###        
    return structures
    
def find_structures_of_given_level(substructure_name, cutofflevel, structures):
    '''Helper function to quickly give progeny of a structure to a given level (note will not give levels higher, i.e. level2 structures will not be shown with level3 input)
    '''
    G = nx.Graph()
    
    #make nodes consisting of structure names and having structures objects stored
    for struct in structures:
        G.add_node(struct.name)
        G.node[struct.name]['structure_object'] = struct #store objects
        if len(struct.children[0]) > 0 :
            for child in struct.children[0]:
                ###adding if statement to in case structure list has been previously pruned!
                if child[1] in [xx.name for xx in structures]:
                    G.add_edge(struct.name, str(child[1]))

        
    #if remove_base_structures == True:
    #    # remove nodes that have no kids:
    #    outdeg = G.out_degree() #get edges pointing out of node
    #    to_remove = [n for n in outdeg if outdeg[n] == 0] #remove nodes with no kids
    #    G.remove_nodes_from(to_remove)
    
    # filter nodes using substructure_name:
    substruct = [x for x in structures if x.name == substructure_name][0]; nodelist=[str(substruct.name)]
    [nodelist.append(str(x[1])) for x in substruct.progeny if x[1] in [xx.name for xx in structures]]
    #[nodelist.append(str(x[1])) for x in substruct.progeny]
    nodelist = list(set(nodelist)); nodelist.sort() #done to remove duplicates

    #filter based on ndegree of separation from node:
    levels = [nx.shortest_path_length(G, substructure_name, node) for node in nodelist] #distance from center node; can use nx.shortest_path to find nodes     
    itemstoremove = [(xx <= cutofflevel) for xx in levels]
    nodelist = list(compress(nodelist, itemstoremove))
    return nodelist
    

def child_remover(structures, ann, verbose=False):
    '''helper function to remove childless strucutres not represented in annotated ABA file
    '''        
    #initialize new list
    nstructures = []
    rm = 0    
    tick = 0
    
    #find unique pixel values in annotation file
    unique_pixel_values = list([int(xx) for xx in np.unique(ann)])
    
    # walk through list, see if pixel id is present in atlas
    for struct in structures:
        tick+=1
        #if no children:
        if len(struct.children[0]) == 0:

            if int(struct.idnum) in unique_pixel_values:  #struc[3] =  ID
                nstructures.append(struct)
            else:
                rm += 1   
        else:
            nstructures.append(struct)
        if verbose: 
            if tick % 50 == 0:
                sys.stdout.write('  {} of {}\n'.format(tick, len(structures)))
    if verbose: sys.stdout.write('{} childless structures removed, as they are not represented in annotated ABA file\n'.format(rm))
    
    return nstructures


def structure_index(structure_id, structurelist):
    '''Function to provide structure index
    '''
    index=[indx for indx, x in enumerate(structurelist) if x.idnum == structure_id]    
    return index[0]


def make_network(structures, substructure_name = 'root', svlocname = None, graphstyle = 'twopi', showcellcounts = True, cutofflevel=10, font_size = 12, show = False):
    '''use networkx to make network of structures, and each structure has the object in its dictionary
    
    _______
    Inputs:
        structures = list of structure objects generated using 
        substructure_name = str (optional) plot only substructure and progeny; e.g. 'Thalamus'
    
    '''
       
    #initialize network
    #G = nx.DiGraph()
    G = nx.Graph()
    
    #make nodes consisting of structure names and having structures objects stored
    for struct in structures:
        G.add_node(struct.name, data=True) #added data=True might break functionality
        G.node[struct.name]['structure_object'] = struct #store objects
        if len(struct.children[0]) > 0 :
            for child in struct.children[0]:
                ###adding if statement to in case structure list has been previously pruned!
                if child[1] in [xx.name for xx in structures]:
                    G.add_edge(struct.name, str(child[1]), weight = 1) #added weight 1, might break functionality

        
    #if remove_base_structures == True:
    #    # remove nodes that have no kids:
    #    outdeg = G.out_degree() #get edges pointing out of node
    #    to_remove = [n for n in outdeg if outdeg[n] == 0] #remove nodes with no kids
    #    G.remove_nodes_from(to_remove)
    
    # filter nodes using substructure_name:
    substruct = [x for x in structures if x.name == substructure_name][0]; nodelist=[str(substruct.name)]
    [nodelist.append(str(x[1])) for x in substruct.progeny if x[1] in [xx.name for xx in structures]]
    #[nodelist.append(str(x[1])) for x in substruct.progeny]
    nodelist = list(set(nodelist)); nodelist.sort() #done to remove duplicates

    #filter based on ndegree of separation from node:
    levels = [nx.shortest_path_length(G, substructure_name, node) for node in nodelist] #distance from center node; can use nx.shortest_path to find nodes     
    itemstoremove = [(xx <= cutofflevel) for xx in levels]
    nodelist = list(compress(nodelist, itemstoremove))

    #add cell counts:    
    if showcellcounts == True:    
        labelswcounts = [(items, xx.cellcount_progeny) for items in nodelist for xx in structures if items == xx.name]        
        newlabels = {}; [newlabels.update([(labelswcounts[xx][0], '{} ({})'.format(labelswcounts[xx][0], labelswcounts[xx][1]))]) for xx in range(len(labelswcounts))] #change names to be 'name (cellcount)'
        G=G.subgraph(nodelist)     
        G = nx.relabel_nodes(G, newlabels) #apply to new labels

        #change node size to reflect counts    
        node_size = [50 + int(xx[xx.rfind('(')+1:xx.rfind(')')]) for xx in G.nodes()]
        node_size = [min(x, 1250) for x in node_size] #threshold to prevent too large nodes

    else:
        G=G.subgraph(nodelist)     
        node_size = 100


    #other features:
    node_color=range(len(G)) #
    #edge_color = range(len(G.edges()))
    alpha = .2 #default is 1
        
    
    #draw graph    'twopi'
    pos = graphviz_layout(G, graphstyle)  #styles=['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']; pos = graphviz_layout(G, types[2])
    
    #nx.draw_networkx(G, pos, node_color = node_color, alpha = alpha)
    #nx.draw_networkx(G, pos, node_size = node_size, node_color = node_color, alpha = alpha, font_size = 8)
    if show != False:    
        nx.draw_networkx(G, pos, node_size = node_size, alpha = alpha, font_size = font_size, node_color = node_color, node_cmap = plt.cm.jet, figsize=(1000,1000))    
    #nx.draw_networkx(G, pos, node_size = node_size, alpha = alpha, font_size = 8, node_color = node_color, node_cmap = plt.cm.jet, edge_color = edge_color, edge_cmap = plt.cm.Blues)
    if svlocname != None:
        plt.savefig(svlocname, dpi=300)
        sys.stdout.write('Saved image as: {}'.format(svlocname))
    return G
   
##################################################################
##################################################################
#########################single radial graph######################
##################################################################
##################################################################
##################################################################   

##FROM http://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart
import numpy as np
#import seaborn as sns # improves plot aesthetics


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=8):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.25,0.25,0.4,0.4],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables, size = 12, rotation = 'horizontal')
                                
        #[txt.set_rotation(angle-90) for txt, angle  #TP REMOVED
         #    in zip(text, angles)] #TP REMOVED
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            #ax.set_rgrids(grid, labels=gridlabel, #TP REMOVED
            #             angle=angles[i]) #TP REMOVED
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


# plotting
def make_complex_radar_charts(excelfl = None, structures = None, substructure_name = 'Thalamus', levels = 3, show = True, svlocname = None, nametype = 'name'):
    '''function to generate polar plot
    
    Inputs
    ---------------
    excelfl = excel file that is outputed from software giving ABA structure cell counts
    structures = list of structure objects generated from this software using ABA
    substructure_name = ABA name to display progeny of a given level
    level = level of substructure progeny of to display
    show = True: display image; False: Do not display image
    svlocname = name, extension and location to save file
    nametype = 'name': display names; 'acronym' display acronyms
    '''
 
    #one of the two needs to be defined
    assert(not(excelfl == None and structures == None))     
 
     #generate objects:
    if excelfl != None:        
        structures = make_structure_objects(excelfl)


    #generate list of structures and counts
    var_count = variable_count(structures, substructure_name, levels, nametype = nametype)
         
    #variable list and data (count) list:
    variables, data = zip(*var_count)
  
    #set range:
    ranges = [(0., float(max(data)+100)) for x in range(len(data))]
  
    #make fig
    fig1 = plt.figure(figsize=(20, 20))
    radar = ComplexRadar(fig1, variables, ranges)
    radar.plot(data)
    radar.fill(data, alpha=0.2)
    
    #add title
    #########################################################################UNFINISHED
    #FIXME: need to add title
    #FIXME: need to add if statement: if acronyms then legend = acronym = name pair
    #########################################################################UNFINISHED
    if show != False:    
        plt.show()
    if svlocname != None:
        plt.savefig(svlocname)
    return

##################################################################
##################################################################
#########################multi radial graphs######################
##################################################################
##################################################################
##################################################################   



"""
From: http://matplotlib.org/examples/api/radar_chart.html
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
#import seaborn as sns # improves plot aesthetics

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize = 10):
            self.set_thetagrids(np.degrees(theta), labels, size = fontsize)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def make_radial_plot(formatted_data, brainnames, add_title = False, legend = False, labelfont = 10):
    #FIXME: need docstring
    N = len(formatted_data[0]) #number of brain_structures
    theta = radar_factory(N, frame='polygon')
    
    #find range to scale radial axis to    
    mx = max([max([x[i] for x in formatted_data[1][1]]) for i in range(len(formatted_data[1][1][0]))]) #find maximum at every indice
    
    #set data
    data = list(formatted_data)
    spoke_labels = data.pop(0)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y', 'g'] ##FIXME:
    
    # Plot the data on separate axes
    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([0.2, 0.4, 0.6, 0.8]) #FIXME: need to rescale this
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        #ax.set_varlabels(spoke_labels, fontsize = 50)#labelfont)

    # add legend relative to top-left plot
    plt.subplot(2, 2, 1)
    labels = tuple(brainnames) #('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = plt.legend(labels, loc=(0.9, .9999), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='medium')

    plt.figtext(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
                ha='center', color='black', weight='bold', size='large')
    plt.show()
    return
    
def subplot_radial_graph(formatted_data, colors, brainnames, fig, n, dim, add_title = None, titlefont = False, subtitlefont = False, legendfont = False, labelfont = False, verbose = False):
    
    if verbose: sys.stdout.write('\n\n{}\n\n'.format(formatted_data)); sys.stdout.flush()
    
    N = len(formatted_data[0]) #number of brain_structures
    theta = radar_factory(N, frame='polygon')
    
    #find range to scale radial axis to    
    mx = max([max([x[i] for x in formatted_data[1][1]]) for i in range(len(formatted_data[1][1][0]))]) #find 'global' max
    
    #set data and sub_axis
    ax = fig.add_subplot(dim, dim, n + 1, projection='radar')    
    data = list(formatted_data)
    spoke_labels = data.pop(0)
    
    #fonts:
    if not titlefont: titlefont = 30,
    if not subtitlefont: subtitlefont = 25
    if not legendfont : legendfont = 25
    if not labelfont : labelfont = 25
    
    # Plot the data on separate axes
    for n, (title, case_data) in enumerate(data):
        #ax = fig.add_subplot(1, 1, 1, projection='radar') #ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([1, 10, 100, 1000]) #plt.rgrids([0.2, 0.4, 0.6, 0.8]) #FIXME: first number needs to be >0 or will kick error
        ax.set_title(title, weight='bold', fontsize=subtitlefont, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.15)
        #ax.set_varlabels(spoke_labels, fontsize = labelfont) #FIXME: fix

    #add title to image    
    if add_title != None:   
        labels = tuple(brainnames) #('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
        legend = plt.legend(labels, loc=(0.9, .9999), labelspacing=0.1)
        plt.setp(legend.get_texts(), fontsize=legendfont)
        plt.figtext(0.5, 0.965, add_title,
                ha='center', color='black', weight='bold', size=titlefont)
        #plt.show()         
       
    return

def make_grouped_radial_plot(inputdata, colors, titlefont = False, subtitlefont=False, legendfont=False, labelfont=False, save = False, title=False, add_acronym_name_pairs = False, acronym_name_pair_font = 14):
    '''Function to make many input graphs, adapted from: http://matplotlib.org/examples/api/radar_chart.html
    ______    
    Inputs
        inputdata = particularly formatted data generated by generate_multi_radial_plots_using_levels function
        title = name to appear on figure
        save = svlocname--- 'path/name.png'; if none does not save
        add_acronym_name_pairs = (optional) acronym: name pairs used to annotate figure
        acronym_name_pair_font = (optional) size of font
        titlefont =  (optional) size of font for title
        subtitlefont = (optional) size of structure labels around radial graph
        legendfont = (optional) size of legend font
    '''
    
    #make graph:
    dim = int(ceil(sqrt(len(inputdata))))
    fig = plt.figure(figsize=(dim*10, dim*10))
    fig.subplots_adjust(wspace=0.25, hspace=0.35, top=0.85, bottom=0.05)
    
    if not title: title = None
    
    #make subplots using single_radial_plot helper function
    for n in range(len(inputdata)):        
                
        if n == 0 :
            #make radial graphs
            formatted_data, brainnames = inputdata[n]
            subplot_radial_graph(formatted_data, colors, brainnames, fig, n, dim, add_title = title, titlefont = titlefont, subtitlefont = subtitlefont, legendfont = legendfont)
        else: 
            #make radial graph and add legend
            formatted_data, brainnames = inputdata[n]
            subplot_radial_graph(formatted_data, colors, brainnames, fig, n, dim, add_title = None, titlefont = titlefont, subtitlefont = subtitlefont, legendfont = legendfont)
    
    #FIXME: WRAP NOT FUNCTIONAL
    if add_acronym_name_pairs: fig.text(0.02, 0.1, add_acronym_name_pairs, fontsize = acronym_name_pair_font) #, wrap=True) #wrap not functional anymore??
        
    if save: plt.savefig(save, dpi=450)
            
    return



def generate_multi_radial_plots_using_levels(excelfls, substructure_name_levels_list = False, title = False, nametype='name', svlocname = False, remove_childless_structures_not_repsented_in_ABA = False, ann_pth=None, colorlst = False, acronym_name_pair_font = 14, titlefont = 20, subtitlefont = 16, legendfont = 14, labelfont = 20, excelfilekeyword=False):
    '''Function to generate multiple plots
    _______    
    Inputs:
        excelfls = list or dictionary of excelfls. If dictionary the keys will be the name associated with the file
        substructure_name_levels_list = [['Thalamus', 2], ['Thalamus', 3], ['Cerebellum', 2], ['Cerebellum', 3], ['Cerebral cortex', 2], ['Cerebral cortex', 3]]
        nametype-- name: returns name of structure; acronym: returns acronym
        title: name to appear on figure
        svlocname: 'path/name.png'; if none does not save
        *args = list of .xlsx files
        excelfilekeyword = (optional) keyword to use for searching; 'cellch' or 'injch' suggested
    '''
    ## testing:
    #excelfls = ['/home/wanglab/wang/pisano/tracing_output/l7cre_ts/ch00_l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_stuctures_table.xlsx', '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/l7_ts05_20150929/ch00_l7_ts05_20150929_488w_647_200msec_z3um_1hfds_stuctures_table.xlsx']

    #deal with FALSES, cannot use 'none' because of *args
    if substructure_name_levels_list == False:
        substructure_name_levels_list=[['Cerebellum', 2], ['Cerebellum', 3], ['Thalamus', 2], ['Thalamus', 3], ['Cerebral cortex', 2], ['Cerebral cortex', 3]]
    if nametype == False:
        nametype = 'name'
        
    #ensure all items in list are strings
    assert(all([type(x)==str for x in excelfls]))

    #generate dictionary of key=excelfl pth value=structure class list:
    sys.stdout.write('Generating structure objects for each excel file, this takes ~1 minute/file\n')    
    structdct = {}
        
    #determine if input was a list or dictionary
    if type(excelfls) == list:
        for excelfl in excelfls:
            
            #optional filtering for cellch vs injch
            if not excelfl[-4:] == '.xls': 
                if excelfilekeyword: excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx and excelfilekeyword in xx][0]            
                else:
                    excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx][0]             
            
            # make structures class lst
            if remove_childless_structures_not_repsented_in_ABA == False:
                structures = make_structure_objects(excelfl) 
                
            elif remove_childless_structures_not_repsented_in_ABA != False:
                    if ann_pth == None:
                        ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
                    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)            
            #set structure class lst
            structdct[excelfl[:excelfl.rfind('/')][excelfl[:excelfl.rfind('/')].rfind('/')+1:]] = structures #brainname:structure pair    
            sys.stdout.write('  -Structure objects successfully made for:\n     {}\n'.format(excelfl[excelfl.rfind('/'):]))        
            
    elif type(excelfls) == dict or type(excelfls) == collections.OrderedDict:
        for brainname, excelfl in excelfls.iteritems():
            
            #optional filtering for cellch vs injch
            if not excelfl[-4:] == '.xls': 
                if excelfilekeyword: excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx and excelfilekeyword in xx][0]            
                else:
                    excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx][0]  
            # make structures class lst
            if remove_childless_structures_not_repsented_in_ABA == False:
                structures = make_structure_objects(excelfl) 
                
            elif remove_childless_structures_not_repsented_in_ABA != False:
                    if ann_pth == None:
                        ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
                    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)            

            #set structure class lst            
            structdct[brainname] = structures #brainname:structure pair    
            sys.stdout.write('  -Structure objects successfully made for:\n     {}\n'.format(excelfl[excelfl.rfind('/'):]))        
            
    sys.stdout.write('Structure objects successfully made for {} excel files\n'.format(len(structdct)))        
        

    #cycle through substructure_name_levels_list
    sys.stdout.write('Generating dictionaries for structure_lvl of cell counts for subregions per brain...\n')        
    tmpdct={}

    #for each of the excel fls determine child cell counts for items in substructure_name_levels_list:                
    for brainname, structures in structdct.iteritems():
        
        #for each in substructure_name_levels_list
        for pair in substructure_name_levels_list:

            #generate list of structures and counts
            substructure_name, levels = pair
            var_count = variable_count(structures, substructure_name, levels, nametype = nametype)
            #print pair,brainname, var_count[0]
            #store in dictionary:    
            try:
                tmpdct['{} Level {}'.format(substructure_name, levels)].update(dict([(brainname, var_count)]))
            except KeyError:
                tmpdct['{} Level {}'.format(substructure_name, levels)] = dict([(brainname, var_count)])

    sys.stdout.write('Done, formatting data, then making graphs...')            

    #format data: for each substructure_name_levels_list populate list    
    inputdata=[]; allstructs = []
    #this particular 'for loop' is done to keep items in substructure_name_levels_list order
    for structurename in ['{} Level {}'.format(xx[0], xx[1]) for xx in substructure_name_levels_list]:
        #walk through and get structure counts
        brainnames = []; [brainnames.append(nm) for nm in tmpdct[structurename].iterkeys()]

        structs = [struc[0] for struc in tmpdct[structurename].values()[0]]; allstructs.append(structs) #allstructs used for acronym:naming pairing
        
        counts = [[count[1] for count in countlst] for countlst in tmpdct[structurename].itervalues()]        

        #finish formatting data
        formatted_data = [structs, [structurename, counts]]

        #make_radial_plot(formatted_data, brainnames) 
        inputdata.append([formatted_data, brainnames])

    ###################### 
    if nametype == 'acronym':
        acronyms = list(set([xx for x in allstructs for xx in x]))
        add_acronym_name_pairs = ['{}: {}'.format(acronym, xx.name) for acronym in acronyms for xx in structures if xx.acronym == acronym]
    else:
        add_acronym_name_pairs = False
        
    #make graph:
    if not colorlst: colorlst = ['b', 'r', 'g', 'm', 'y', 'g']
    make_grouped_radial_plot(inputdata, colorlst, save = svlocname, title=title, add_acronym_name_pairs = add_acronym_name_pairs, acronym_name_pair_font = acronym_name_pair_font, titlefont = titlefont, subtitlefont = subtitlefont, legendfont = legendfont, labelfont = labelfont) 
            
    return


def variable_count(structures, substructure_name, levels, nametype='name'):
    '''helper function to return list of progeny of a substructure of a given level 
    _______
    Inputs:
        structures = list of stucture objects
        substructure_name = ABA name of structure to look for progeny of        
        levels = level of progeny of substructure_name to list 
        nametype = name: returns name of structure; acronym: returns acronym

    '''    
    #make network x structures
    G = make_network(structures, substructure_name = substructure_name, graphstyle = 'twopi', showcellcounts = False, cutofflevel=levels, font_size = 11, show = False)        
    
    #find structures at level AND structures at higher levels without children; then get cell counts
    if nametype == 'name':    
        lst = [(xx.name, xx.cellcount_progeny) for x in G.nodes() if nx.shortest_path_length(G, substructure_name, x) == levels for xx in structures if xx.name == x] #structures at level
        [lst.append((xx.name, xx.cellcount_progeny)) for x in G.nodes() if nx.shortest_path_length(G, substructure_name, x) < levels for xx in structures if xx.name == x and len(xx.children[0]) == 0] #'higher structures' that don't have kids
    elif nametype == 'acronym':
        lst = [(xx.acronym, xx.cellcount_progeny) for x in G.nodes() if nx.shortest_path_length(G, substructure_name, x) == levels for xx in structures if xx.name == x] #structures at level
        [lst.append((xx.acronym, xx.cellcount_progeny)) for x in G.nodes() if nx.shortest_path_length(G, substructure_name, x) < levels for xx in structures if xx.name == x and len(xx.children[0]) == 0] #'higher structures' that don't have kids
    else:
        print ('Error: nametype is not "name" or "acronym"')
    #variable list and data (count) list:
    #variables, count = zip(*lst)
    return lst #variables, count 


def generate_multi_radial_plots_using_lists(excelfls, title_substructure_list = False, title = False, nametype='name', svlocname = False, remove_childless_structures_not_repsented_in_ABA = False, ann_pth=None, colorlst = False, acronym_name_pair_font = 14, titlefont = 20, subtitlefont = 16, legendfont = 14, labelfont = 20, excelfilekeyword=False):
    '''Function to generate multiple plots using lists
    _______    
    Inputs:
        excelfls = list or dictionary of excelfls. If dictionary the keys will be the name associated with the file
        title_substructure_list = list where first entry is graph title followed by structures [['Midbrain', 'Structure1', 'Structure2', 'Structure3'], ['Thalamus', Structure1, Structure2....] ....]; Structures can be names or acronyms
        nametype-- name: returns name of structure; acronym: returns acronym
        title: name to appear on figure
        svlocname: 'path/name.png'; if none does not save
        *args = list of .xlsx files
        font = size of text
        acronym_name_pair_font, titlefont, subtitlefont, legendfont, labelfont: optional font sizes
        excelfilekeyword = (optional) keyword to use for searching; 'cellch' or 'injch' suggested
    '''
    ## testing:
    #excelfls = ['/home/wanglab/wang/pisano/tracing_output/l7cre_ts/ch00_l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_stuctures_table.xlsx', '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/l7_ts05_20150929/ch00_l7_ts05_20150929_488w_647_200msec_z3um_1hfds_stuctures_table.xlsx']

    #deal with FALSES, cannot use 'none' because of *args
    if title_substructure_list == False:
        title_substructure_list=[['Cerebellum Level 2', 'IP', 'DN', 'FN', 'VERM', 'HEM'], ['Cerebellum Level 3', 'UVU','DEC','PYR','PFL','LING','PRM','CUL','COPY','AN','CENT','FL','SIM','NOD','FOTU','DN','IP','FN'], ['Thalamus Level 2', 'SPA','EPI','MTN','VENT','GENv','PP','GENd','ILM','LAT','ATN','MED','SPF','RT'] , ['Thalamus Level 3', 'LGv','LP','SMT','CL','SPFp','SPFm','IAD','PT','PO','SGN','VM','MG','IGL','RE','SubG','PF','IMD','RH','MH','PCN','CM','IAM','VAL','LD','AV','AD','PR','MD','POL','LGd','VP','PVT','AM','LH','RT','PP','SPA'], ['Cerebral cortex Level 2','OLF', 'CLA', 'EP', 'HPF', 'Isocortex', 'LA', 'BLA', 'PA', 'BMA'] , ['Cerebral cortex Level 3', 'EPv','ACA','VISC','PAA','EPd','FRP','ILA','NLOT','RSP','PL','TR','MO','BLAp','AON','HIP','COA','VIS','TEa','ECT','GU','ORB','SS','MOB','DP','PERI','PTLp','RHP','AOB','AI','PIR','AUD','BLAa','BLAv','TT','PA','LA','CLA']]
    if nametype == False:
        nametype = 'name'
        

    #ensure all items in list are strings
    assert(all([type(x)==str for x in excelfls]))

    #generate dictionary of key=excelfl pth value=structure class list:
    sys.stdout.write('Generating structure objects for each excel file, this takes ~1 minute/file\n')    
    structdct = {}
        
    #determine if input was a list or dictionary
    if type(excelfls) == list:
        for excelfl in excelfls:
            
            #optional filtering for cellch vs injch
            if not excelfl[-4:] == '.xls': 
                if excelfilekeyword: excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx and excelfilekeyword in xx][0]            
                else:
                    excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx][0]            
            
            # make structures class lst
            if remove_childless_structures_not_repsented_in_ABA == False:
                structures = make_structure_objects(excelfl) 
                
            elif remove_childless_structures_not_repsented_in_ABA != False:
                    if ann_pth == None:
                        ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
                    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)            
            #set structure class lst
            structdct[excelfl[:excelfl.rfind('/')][excelfl[:excelfl.rfind('/')].rfind('/')+1:]] = structures #brainname:structure pair    
            sys.stdout.write('  -Structure objects successfully made for:\n     {}\n'.format(excelfl[excelfl.rfind('/'):]))        
            
    elif type(excelfls) == dict or type(excelfls) == collections.OrderedDict:
        for brainname, excelfl in excelfls.iteritems():

            #optional filtering for cellch vs injch
            if not excelfl[-4:] == '.xls': 
                if excelfilekeyword: excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx and excelfilekeyword in xx][0]            
                else:
                    excelfl = [xx for xx in listdirfull(excelfl) if '.xls' in xx][0]            

            # make structures class lst
            if remove_childless_structures_not_repsented_in_ABA == False:
                structures = make_structure_objects(excelfl) 
                
            elif remove_childless_structures_not_repsented_in_ABA != False:
                    if ann_pth == None:
                        ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
                    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)            

            #set structure class lst            
            structdct[brainname] = structures #brainname:structure pair    
            sys.stdout.write('  -Structure objects successfully made for:\n     {}\n'.format(excelfl[excelfl.rfind('/'):]))        
            
    sys.stdout.write('Structure objects successfully made for {} excel files\n'.format(len(structdct)))        
        
    #cycle through title_substructure_list
    sys.stdout.write('Generating dictionaries for structure_lvl of cell counts for subregions per brain...\n')        
    tmpdct={}

    #for each of the excel fls determine child cell counts for items in title_substructure_list:                
    for brainname, structures in structdct.iteritems():
        

        #for each in title_substructure_list
        for lst in title_substructure_list:

            var_count=[] ###FIXME:
            subtitle = lst[0]
            for substructure_name in lst[1:]:
                #struct_obj = [xx for xx in structures if xx.name == substructure_name or xx.acronym == substructure_name][0]
                ####STOPPED HERE - need  ('Peripeduncular nucleus', 4), ('Subparafascicular area', 7)] but acceptable with acronym, or name
                #generate list of structures and counts
                if nametype == 'name':
                    [var_count.append((xx.name, xx.cellcount_progeny)) for xx in structures if xx.name == substructure_name or xx.acronym == substructure_name] #('Subparafascicular area', 7)
                elif nametype == 'acronym':
                    [var_count.append((xx.acronym, xx.cellcount_progeny)) for xx in structures if xx.name == substructure_name or xx.acronym == substructure_name] #('Subparafascicular area', 7)
                #print pair,brainname, var_count[0]
                #store in dictionary:    
            try:
                tmpdct[subtitle].update(dict([(brainname, var_count)]))
            except KeyError:
                tmpdct[subtitle] = dict([(brainname, var_count)])

    sys.stdout.write('Done, formatting data, then making graphs...')     

    #format data: for each title_substructure_list populate list    
    inputdata=[]; allstructs = []
    #this particular 'for loop' is done to keep items in title_substructure_list order
    for structurename in [xx[0] for xx in title_substructure_list]:
        #walk through and get structure counts
        brainnames = []; [brainnames.append(nm) for nm in tmpdct[structurename].iterkeys()]

        structs = [struc[0] for struc in tmpdct[structurename].values()[0]]; allstructs.append(structs) #allstructs used for acronym:naming pairing
        
        counts = [[count[1] for count in countlst] for countlst in tmpdct[structurename].itervalues()]        

        #finish formatting data
        formatted_data = [structs, [structurename, counts]]

        #make_radial_plot(formatted_data, brainnames) 
        inputdata.append([formatted_data, brainnames])

    ###################### 
    if nametype == 'acronym':
        acronyms = list(set([xx for x in allstructs for xx in x]))
        add_acronym_name_pairs = ['{}: {}'.format(acronym, xx.name) for acronym in acronyms for xx in structures if xx.acronym == acronym]; add_acronym_name_pairs.sort()
    else:
        add_acronym_name_pairs = False
        
    #make graph:
    if not colorlst: colorlst = ['b', 'r', 'g', 'm', 'y', 'g']
    make_grouped_radial_plot(inputdata, colorlst, save = svlocname, title=title, add_acronym_name_pairs = add_acronym_name_pairs, acronym_name_pair_font = acronym_name_pair_font, titlefont = titlefont, subtitlefont = subtitlefont, legendfont = legendfont, labelfont = labelfont) 
    sys.stdout.write('\n\nGraph saved as {}'.format(svlocname))
    return


def add_progeny_counts_at_each_level(df, df_pth = '/jukebox/wang/pisano/Python/lightsheet/supp_files/ls_id_table_w_voxelcounts.xlsx', ann_pth = '/jukebox/wang/pisano/Python/atlas/annotation_sagittal_atlas_20um_iso.tif'):
    '''
    '''
    #make structures
    structures = make_structure_objects(df_pth, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    
    #make copy of df so not to count things multiple times often
    ddf = df.copy()
    ddf[:] = 0
    
    #now do prog
    for s in structures:
        if len(s.progeny)!=0:
            #break
            prog = [xx.name for xx in s.progeny]
            s_vals = df[df.index==s.name].values
            prog_df_vals = df[df.index.isin(prog)].sum(0).values
            sums = s_vals + prog_df_vals
            ddf[ddf.index==s.name] = sums
        else:
            ddf[ddf.index==s.name] = df[df.index==s.name].values
            
    return ddf        

if __name__ == '__main__':    
    #sns.reset_orig() #w/o you get black and white graphs
    #testing
    excelfl = '/home/wanglab/wang/pisano/Python/lightsheet/supp_files/sample_cell_count_output.xlsx'
    #structure_id= 549 #Thalamus's id NOT atlas_id
    ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
    
    #load    
    df = pd.read_excel(excelfl)

    #generate objects:
    from tools.analysis.network_analysis import make_structure_objects
    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    #thal = [xx for xx in structures if 'Thalamus' == xx.name][0]
    
    #make networkx graphs    
    make_network(structures, substructure_name = 'Thalamus', graphstyle = 'twopi', showcellcounts = True, cutofflevel=3, font_size = 11, show = True)        
    make_network(structures, substructure_name = 'Cerebral cortex', graphstyle = 'twopi', showcellcounts = True, cutofflevel=3, font_size = 11, show = True)
    make_network(structures, substructure_name = 'Cerebellum', graphstyle = 'twopi', showcellcounts = True, cutofflevel=3, font_size = 13, show = True)    

    #save networkx graphs
    svlocname = '/home/wanglab/wang/pisano/Submissions+Presentations/Presentations/Current_pres/networkx_no_childless_structures'
    make_network(structures, substructure_name = 'Thalamus', svlocname = svlocname + '/_Thalamus_lvl2.png', graphstyle = 'twopi', cutofflevel=2, show = True)
    #[make_network(structures, substructure_name = 'Thalamus', svlocname = svlocname + '/_Thalamus_lvl2_{}.png'.format(xx), graphstyle = xx, cutofflevel=2, show = True) for xx in ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']]
    make_network(structures, substructure_name = 'Cerebral cortex', svlocname = svlocname+'/_Cerebral_cortex_lvl3.png', graphstyle = 'twopi', cutofflevel=3, show = True)
    make_network(structures, substructure_name = 'Cerebellum', svlocname = svlocname+'/_Cerebellum_lvl3.png', graphstyle = 'twopi', cutofflevel=3, show = True)

    import seaborn as sns    #do not move - it messes up make_network
    ##########################
    #make single radial graphs
    excelfl = '/home/wanglab/wang/pisano/tracing_output/l7cre_ts/ch00_l7cre_ts01_20150928_005na_z3um_1hfds_488w_647_200msec_5ovlp_stuctures_table.xlsx'
    ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015.nrrd'
    svlocname = '/home/wanglab/wang/pisano/Submissions+Presentations/Presentations/Current_pres/Radial_plots/examples'
    svlocname = '/home/wanglab/wang/pisano/Submissions+Presentations/Presentations/Current_pres/Radial_plots/children_removed/acronyms'
    structures = make_structure_objects(excelfl)
    structures = make_structure_objects(excelfl, remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    nametype = 'acronym' #'name'
    
    #thalamus
    make_complex_radar_charts(structures = structures, substructure_name = 'Thalamus', levels = 2, show = False, svlocname = svlocname + '/radialplot_thalamus_lvl2.png', nametype = nametype)    
    make_complex_radar_charts(structures = structures, substructure_name = 'Thalamus', levels = 3, show = False, svlocname = svlocname + '/radialplot_thalamus_lvl3.png', nametype = nametype)
    make_complex_radar_charts(structures = structures, substructure_name = 'Thalamus', levels = 4, show = False, svlocname = svlocname + '/radialplot_thalamus_lvl4.png', nametype = nametype)
    #neocortex
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebral cortex', levels = 2, show = False, svlocname = svlocname + '/radialplot_cerebralcortex_lvl2.png', nametype = nametype)    
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebral cortex', levels = 3, show = False, svlocname = svlocname + '/radialplot_cerebralcortex_lvl3.png', nametype = nametype)    
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebral cortex', levels = 4, show = False, svlocname = svlocname + '/radialplot_cerebralcortex_lvl4.png', nametype = nametype)    
    #cerebellum
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebellum', levels = 2, show = False, svlocname = svlocname + '/radialplot_cb_lvl2.png', nametype = nametype)        
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebellum', levels = 3, show = False, svlocname = svlocname + '/radialplot_cb_lvl3.png', nametype = nametype)    
    make_complex_radar_charts(structures = structures, substructure_name = 'Cerebellum', levels = 4, show = False, svlocname = svlocname + '/radialplot_cb_lvl4.png', nametype = nametype)    


