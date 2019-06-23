#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:35:25 2017

@author: tpisano

see: tools.conv_net.mem_map for example usage

"""

from tools.utils.io import chunkit
import multiprocessing as mp
import pandas as pd
import numpy as np
from math import sqrt
import  sys, os, cv2, gc, time


class cell_class:
    """Class to represent a cell
    z,y,x is the location of the determined center ###FOLLOWING NP CONVENTION
    plns=[z,x,y,cntrs] of cell detection ###NOTE THE CONVENTION CHANGE DUE TO CV2"""
    kind='cell'    
    def __init__(self, cellnumber):
        self.cellnumber=cellnumber ##id of cell        
        self.plns={}               #{'00'   :  [z,x,y,cntr]} ; cntrs=x,y; THIS IS BECAUSE OF CV2
        self.center=()             #final z,y,x of center
        self.radius=()             #determined radius
        self.volume=()             #determined volume    
        self.connectedwith=[]      #other cells connected with
        self.activecell=True       #True: cell is represented on previous pln; false: cell's connections have been determined
        
    def add_pln(self, pln): 
        if len(self.plns)==0:
            key='000'
        else:
            key=str(len(self.plns)).zfill(3)
        self.plns[key]=pln
    def remove_pln(self, key):
        del self.plns[key]
    def add_radius(self, r): 
        self.radius=(r)
    def add_center(self, z,y,x): 
        self.center=(z,y,x)
    def add_volume(self, vol):
        self.volume=(vol)
    def add_connection(self, othercell):
        if othercell not in self.connectedwith:
            self.connectedwith.append(othercell)
    def inactivatecell(self):
        self.activecell=False
    def activatecell(self):
        self.activecell=True
    def change_cellnumber(self, newnum):
        self.cellnumber=newnum

def detect_centers_from_contours(rois, cores=5, mxdst=50, verbose = False):
    '''Goal: combine a cell that has been identified on multiple 2d planes.
    take all detected cells centers/contours and load them into a list of lists; sort based on z, then y, then x. Knowing this organization one
    can utilize chunking and find rough ranges first in z, then y, then x to determine range of indices to look for overalpping points. 
    Note: this is a fast (3-5 minutes depending on the number of cells) but VERY memory intensive step. If memory consumption is too much,
    decrease pln_chnk.
    
    rois format:
        each roi is [z,y,x, [contour]]; remeber IMAGEJ has one-based numerics for z plane
        
        
    Note: assumes contours are from CV2 thus convention of XY. If contours are from FIJI/IMAGEJ (YX not XY) then you need to swap XY'''
    
    ###start###
    if verbose: print ('Detecting centers from contours using {} cores'.format(cores))
    try:
        p
    except NameError:
        p=mp.Pool(cores)    
    if verbose: sys.stdout.write('\n   mp version {}'.format(mp.__version__)); sys.stdout.flush()

    #sort by z:
    rois = sorted(rois, key=lambda a: a[0])    
    
    ###create dataframe
    cells=pd.DataFrame([(rois[xx][0], rois[xx][1], rois[xx][2], xx) for xx in range(len(rois))], columns=list(('z', 'y', 'x', 'contourindex')))
    if verbose: sys.stdout.write('\n   dataframe made. Starting 3d grouping...'); sys.stdout.flush()
    
    ########iterate through cntrs; work through each zpln looking for overlap in next zpln, if overlap add to current cell, if no overlap make new active cell; if cell didn't have overlap in this plane make it inactive
    zplns=list(set(cells['z'])); zplns.sort()
    activecells={}; inactivecells={};
    for i in zplns:    
        if i in zplns:
            ztime=time.time()        
            ##for each cell, find cells that fall within mxdst in x and y on the plane above
            cells_in_pln=cells.loc[cells.z == i]                         
            iterlst=[]; [iterlst.append((mxdst, cells, activecells, inactivecells, cells_in_pln, rois, core, cores)) for core in range(cores)]           
            lst = p.starmap(cell_connected_check, iterlst); del iterlst; 
            activecells={}; [activecells.update(xx[0]) for xx in lst]
            [inactivecells.update(xx[1]) for xx in lst]; del lst
            if verbose: 
                sys.stdout.write('\n   {} seconds for zpln {}, cells in plane {}, active cells {}'.format(round((time.time()-ztime), 2), i, len(cells_in_pln), len(activecells))); sys.stdout.flush()
            gc.collect()

    celldct={}
    for keys,values in inactivecells.iteritems():
        tmp=cell_class(keys); [tmp.add_connection(xx) for xx in values]
        celldct[keys]=tmp
    ###add z plns that each cell spans; NOTE THE PLANES ARE IN XY FROM CV2
    for keys, values in celldct.iteritems():
        values.add_pln(rois[keys])        
        [values.add_pln(rois[xx]) for xx in values.connectedwith]        
    ###find centers
    tplst=find_radius_and_center(celldct) ###numbers are in PIXELS

    if verbose: sys.stdout.write('\n\n***{} centers found***\n\n'.format(len(tplst))); sys.stdout.flush()
    
    p.terminate()
    return tplst
    
  
def cell_connected_check(mxdst, cells, activecells, inactivecells, cells_in_pln, cell_lst_srtd, core, cores):
    ####determine chunk size based on num of cores
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")      
    chnkrng=chunkit(core, cores, cells_in_pln)
    ####process    
    activecells2={}; inactivecells2={}  
    for j in cells_in_pln.index[chnkrng[0]-1:chnkrng[1]]:
        yrn=range(cells.y[j]-mxdst, cells.y[j]+mxdst); xrn=range(cells.x[j]-mxdst, cells.x[j]+mxdst)        
        #cells_above=cells[cells.z == cells.z[j]+1][cells.y.isin(yrn)][cells.x.isin(xrn)]          
        cells_above=cells[(cells.z == cells.z[j]+1) & cells.y.isin(yrn) & cells.x.isin(xrn)]          
        if len(cells_above) == 0: #no other cells within range on pln above, make it a cell and add to inactive list
            try: #try to see if cell was active on last plane, if so place as inactive cell in inactive dct
                tmp=activecells[j]
                inactivecells2[j]=tmp
                del activecells[j];
            except KeyError: ##cell wasn't in previous pln, and thus represented only on a single pln
                tmp=[j]
                inactivecells2[j]=tmp
    ###test if connected to others
        else: #check to see if j is connected to others
            onlycell=True                
            for ii in cells_above.index:
                if connected(ii, j, cell_lst_srtd) > 0:
                    try: #try to see if cell was active on last plane,
                        tmp=activecells[j]; tmp.append(ii)
                        activecells2[ii]=tmp
                        del activecells[j]
                        onlycell=False;
                    except KeyError: #not present in previous active list
                        tmp=[j]
                        activecells2[ii]=tmp
                        onlycell=False
            if onlycell==True: ##if no connections were found then treat it as an independent cell
                try: #try to see if cell was active on last plane, if so place as inactive cell in inactive dct
                    tmp=activecells[j]
                    inactivecells2[j]=tmp
                    del activecells[j]
                except KeyError: ##cell wasn't in previous pln
                    tmp=[j]
                    inactivecells2[j]=tmp
        gc.collect()
    try:
        del activecells; del inactivecells; del tmp; del yrn; del xrn; del cell_lst_srtd; del cells; del cells_in_pln; del cells_above
    except NameError:
        pass
    #print ('Chnkrange {} time taken {} minutes'.format(chnkrng, (time.time()-starttime)/60))
    return activecells2, inactivecells2

def connected(ii, j, cell_lst_srtd):
    '''Simple Function to test for overlap between two contours. Contour convention is XY due to CV2, NP it is YX
    
    Inputs
    ----------------
    ii = first index of list to compare
    j = second index of list to compare
    cell_lst_srtd = list of contours in XY
    '''    
    ii=cell_lst_srtd[ii][3]
    j=cell_lst_srtd[j][3]
    xmin,ymin=np.min(zip(np.min(j, axis=0),  np.min(ii, axis=0)), axis=1) #find minimum and maximum values and only compute across this area to increase efficiency
    xmax,ymax=np.max(zip(np.max(j, axis=0),  np.max(ii, axis=0)), axis=1) ###flipped X and Y's on this and line above because of CV2s convention
    im1=np.zeros((ymax-ymin, xmax-xmin)); cv2.fillPoly(im1, [np.int32(j-[xmin, ymin])], color=(255,255, 255));  #and here flipping of yx on the cv2 part
    im2=np.zeros((ymax-ymin, xmax-xmin)); cv2.fillPoly(im2, [np.int32(ii-[xmin, ymin])], color=(255,255, 255)); #and here flipping of yx on the cv2 part
    if np.argmax(im1*im2)>0:
        del im1, im2, cell_lst_srtd, ymin, xmin, ymax, xmax 
        gc.collect()
        return 1
    else:
        del im1, im2, cell_lst_srtd, ymin, xmin, ymax, xmax
        gc.collect()
        return 0    
        
def find_radius_and_center(celldct):  
    '''inputs celldct consisting of , and x,y,z micron/pixel scales
    Determines center by finding span of cell representation in z and taking the middle as the z center. 
    Determines Radius by taking the average center to contour distance of the center pln (determined from above), if even #plns takes average of two middles plns
    ***********Ouputs are PIXELS currently*********************
    '''
  ################THIS NEEDS TO BE CHECKED INCOMPLETE
    ##find radius in z and in xy, average the two and use this for volume calc
    bg_cell_class_lst1=[]; tick=0
    for num, cell in celldct.iteritems():
        tick+=1
        ###check for duplicate planes, if so remove them
        key_zpln=[[keys, values[0]] for keys, values in cell.plns.iteritems()]
        removelst=[]        
        while len(key_zpln) > 0:        
            i=key_zpln.pop()
            for ii in key_zpln:
                if i[1] == ii[1]:
                    removelst.append(i[0])
        [cell.remove_pln(xx) for xx in removelst]
        ###set center for cells represented on one plane
        if len(cell.plns) == 1:
            z,x,y=cell.plns.values()[0][0:3] #flipping from cv2 to np
            cell.add_center(z,y,x)
            ###find radius by calculating average distance of contours from center            
            cell.add_radius(np.mean([sqrt((x-i[0])**2 + (y-i[1])**2) for i in cell.plns.values()[0][3]]))
        else:###cells on multiple planes
            zyxlst=[(pln[0:3]) for num, pln in cell.plns.iteritems()]; zyxlst.sort()
            zyxcntlst=[(pln) for num, pln in cell.plns.iteritems()]; zyxcntlst.sort()
            #determine center: if odd number of plns, take middle. If even take average between middle two; might need to make ints
            if len(zyxlst)%2==0: #even
                midlst=[zyxlst[int(i)] for i in [len(zyxlst)/2, len(zyxlst)/2-1]];
                z,x,y=tuple((i+j)/2 for i,j in zip(midlst[0], midlst[1]))
                cell.add_center(z,y,x) #converting from cv2 to np
                #cell.add_center(*(tuple((i+j)/2 for i,j in zip(midlst[0], midlst[1])))) ##wrong due to cv2=xy and np=yx
                #z,y,x=cell.plns.values()[0][0:3] #don't need anymore
                cell.add_radius(np.mean([sqrt((x-i[0])**2 + (y-i[1])**2) for aa in [ii[3] for ii in zyxcntlst[int((len(zyxcntlst)/2)-1):int((len(zyxcntlst)/2)+1)]] for i in aa])) ###take middle two and find the average of them
            else: ###take middle of odd
                z,x,y=zyxlst[int(len(zyxlst)/2)+1]
                cell.add_center(z,y,x) #converting from cv2 to np
                ###find radius by calculating average distance of contours from center at CENTER PLANE                
                key=[keys for keys, values in cell.plns.iteritems() if z == values[0]]
                cell.add_radius(np.mean([sqrt((x-i[0])**2 + (y-i[1])**2) for i in cell.plns[key[0]][3]]))
            #cell.add_volume()
        ####apply some filtering here
        bg_cell_class_lst1.append(cell)
    return bg_cell_class_lst1
