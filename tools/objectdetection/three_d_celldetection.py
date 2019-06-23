# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:11:55 2016

@author: wanglab
"""
from __future__ import division
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer
import multiprocessing as mp
import pandas as pd
import numpy as np
from math import sqrt, ceil, pi, floor
import re, sys, os, cv2, gc, shutil, warnings, time, random
import cPickle as pickle
from skimage.external import tifffile
from tools.imageprocessing.preprocessing import resample_par
from tools.imageprocessing.orientation import fix_orientation, fix_contour_orientation, fix_dimension_orientation
import subprocess as sp
import SimpleITK as sitk
from collections import Counter
#%%

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
#%%
def detect_cells_in_3d(jobid, cores=None, mxdst=None, pln_chnk=None, ovlp_plns=None, **kwargs):
    '''Goal: combine a cell that has been identified on multiple 2d planes.
    take all detected cells centers/contours and load them into a list of lists; sort based on z, then y, then x. Knowing this organization one
    can utilize chunking and find rough ranges first in z, then y, then x to determine range of indices to look for overalpping points. 
    Note: this is a fast (3-5 minutes depending on the number of cells) but VERY memory intensive step. If memory consumption is too much,
    decrease pln_chnk.'''###inputs###
    if cores==None:
        cores = 5
    ###inputs    
    kwargs = load_kwargs(**kwargs)
    outdr=kwargs['outputdirectory']
    vols=kwargs['volumes']
    reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]
    celldetect3dfld = reg_vol.celldetect3dfld
    xscl, yscl, zscl = reg_vol.xyz_scale ###micron/pixel  
    cellcoordinatesfld = reg_vol.cellcoordinatesfld
    ###start###
    bigstart=time.time()    
    print ('cores: {}'.format(cores))
    try:
        p
    except NameError:
        p=mp.Pool(cores)    
    print ('mp version {}'.format(mp.__version__))    
    ############################################################################################################
    #######################use regex to sort np files by ch and then by zpln####################################
    ############################################################################################################       
    fl=[f for f in os.listdir(cellcoordinatesfld) if '.npy' in f] #sorted for raw files
    reg=re.compile(r'(.*h+)(?P<ch>\d{2})(.*Z+)(?P<z>\d{4})(.npy)')
    matches=map(reg.match, fl)      
    ##find index of z,y,x,ch in a match str
    z_indx=matches[0].span('z')
    ###determine number of channels
    #chs=[]; [chs.append(matches[i].group('ch')[-2:]) for i in range(len(matches)) if matches[i].group('ch')[-2:] not in chs]  
    cell_chs=[vol for vol in vols if vol.ch_type == 'cellch']
    ###determine max of z channels
    zmx=max([int(matches[i].group('z')) for i in range(len(matches))])
    ############################################################################################################
    #######################parse up jobs to array ##############################################################
    ############################################################################################################       
    if pln_chnk == None:
        pln_chnk=zmx ###for very dense labeling you may need to set this to a smaller number to prevent memory issues
    if mxdst == None:
        mxdst=50
    if ovlp_plns == None: ####number of planes that need to be shared by adjacent jobs
        ovlp_plns=int(pln_chnk) #####each job will consist of 2*pln_chnk, this is to account for cells that lie in the ovlping plns
    plns_per_job=int(pln_chnk + ovlp_plns) 
    ##########################Planes to process#######################################################
    plns=[int(floor(jobid / len(cell_chs)) * pln_chnk) + x for x in range(plns_per_job)];  plns.sort() ###each job coverage.
    valid_plns=plns[:pln_chnk+1]
    if jobid not in range(len(cell_chs)): ###add the previous plane so that job can determine if a cell represented on first valid pln is NOT present on first valid pln-1
        plns.append(min(plns)-1)
    plns.sort(); valid_plns.sort()
    ################cancel jobs if jobid past valid planes######################
    if jobid > len(cell_chs)*int(ceil(zmx/pln_chnk)):
        print ('jobid({}) x pln_chnk({}) x cell_chs({}) is greater than total number of planes({})'.format(jobid, pln_chnk, zmx, len(cell_chs)))        
        p.terminate()
        return    
    ##################run per channel; in this case chs represent only 'signal' chs ##################
    #chs=[xx for xx in kwargs['channels'] if kwargs['regch'] !=xx]
    ch=cell_chs[int(jobid % len(cell_chs))]      
    print('Jobid ({}), starting 3D cell group on {} ({})...'.format(jobid, ch.channel, ch.ch_type))
    lst=[]
    zdct={}
    try:     
        [lst.append(os.path.join(cellcoordinatesfld, ''.join(match.groups()))) for num,match in enumerate(matches) if ch.channel in match.group('ch')] ##make list of all .np files for channel
    except:
        pass
    ######make list of only the pertienent planes#####
    ###find planes that are represented both in cellcoordinates fld and in plns lst         
    chklst=set([npys for pln in plns for npys in lst if str(pln).zfill(4) in npys[-8:-4]]); lst=[x for x in chklst]; lst.sort()
    #########
    srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]])) #sort by z
    print ('loading np files for ch{}, plns {}-{}...'.format(ch.channel, plns[0], plns[-1]))
    for fl in srtd:
        npfl=np.load(fl)
        zdct[int(fl[fl.rfind('Z')+1:fl.rfind('.')])]=npfl                   
    cell_lst=[[z, int(cnt[i, 0][1]), int(cnt[i, 0][2]), cnt[i, 1]] for z,cnt in zdct.iteritems() for i in range(len(cnt))] #lst of [z_center,y_center,x_center, contours]                
    print ('done loading np files, {} initial cells in lst'.format(len(cell_lst)))  
    ##tmp
    #a={}; a.update(dict([('cell_lst_srtd', cell_lst_srtd)])); 
    #pckloc=os.path.join(outdr, 'testtest.p'); pckfl=open(pckloc, 'wb'); pickle.dump(a, pckfl); pckfl.close() 
    ##tmp    
    sys.stdout.write('grouping cells on adjacent planes by proximity of mxdst({}), then checking for pixel overlap...'.format(mxdst))    
    ###create z,y,x,indx(of cell_lst_srtd) in micron scale dataframe
    sys.stdout.write('creating dataframe...')
    cell_lst_srtd=sorted(cell_lst, key=lambda a: (a[0], a[1], a[2])) #sort by z, then y, then x; might not have to do this
    cells=pd.DataFrame([[cell_lst_srtd[i][0], cell_lst_srtd[i][1], cell_lst_srtd[i][2], i] for i in range(len(cell_lst_srtd))], columns=list(('z', 'y', 'x', 'contourindex')))
    sys.stdout.write('dataframe made. Starting 3d grouping...'); sys.stdout.flush()
    ########iterate through cntrs; work through each zpln looking for overlap in next zpln, if overlap add to current cell, if no overlap make new active cell; if cell didn't have overlap in this plane make it inactive
    zplns=list(set(cells['z'])); zplns.sort()
    activecells={}; inactivecells={};
    for i in zplns:    
        if i in valid_plns or i == (valid_plns[0] - 1 ): ###process each z pln in valid planes or in the plane before valid plns and will remove later
            ztime=time.time()        
            ##for each cell, find cells that fall within mxdst in x and y on the plane above
            cells_in_pln=cells.loc[cells.z == i]                         
            iterlst=[]; [iterlst.append((mxdst, cells, activecells, inactivecells, cells_in_pln, cell_lst_srtd, core, cores)) for core in range(cores)]           
            lst=p.map(cell_connected_check, iterlst); del iterlst; 
            activecells={}; [activecells.update(xx[0]) for xx in lst]
            [inactivecells.update(xx[1]) for xx in lst]; del lst
            sys.stdout.write('\n{} minutes for ch {}, zpln {}, cells in plane {}, active cells {}, inactive cells {}'.format(round((time.time()-ztime)/60, 4), ch.channel, i, len(cells_in_pln), len(activecells), len(inactivecells))); sys.stdout.flush()
            print sys.getsizeof(inactivecells)
            gc.collect()
            #pckloc=os.path.join(outdr, 'ch{}_inactivecells_plns{}-{}.p'.format(ch, valid_plns[0], valid_plns[1])); pckfl=open(pckloc, 'wb'); pickle.dump(inactivecells, pckfl); pckfl.close()
            #del inactivecells; inactivecells={}   
        elif len(activecells) != 0: ####don't process cells overlap unless they are cells that extended into overlap
            ztime=time.time()        
            ##for each cell STILL IN ACTIVE LIST*****, find cells that fall within mxdst in x and y on the plane above
            active_cells_in_pln=cells[cells.index.isin(activecells.keys())]
            iterlst=[]; [iterlst.append((mxdst, cells, activecells, inactivecells, active_cells_in_pln, cell_lst_srtd, core, cores)) for core in range(cores)]           
            lst=p.map(cell_connected_check, iterlst); 
            activecells={}; [activecells.update(xx[0]) for xx in lst]
            [inactivecells.update(xx[1]) for xx in lst]; del lst
            sys.stdout.write('\n{} minutes for ch {}, zpln {}, cells in plane {}, active cells {}, inactive cells {}'.format(round((time.time()-ztime)/60, 4), ch.channel, i, len(cells_in_pln), len(activecells), len(inactivecells))); sys.stdout.flush()
            print sys.getsizeof(inactivecells)
            gc.collect()
            #pckloc=os.path.join(outdr, 'ch{}_inactivecells_plns{}-{}.p'.format(ch, valid_plns[0], valid_plns[1])); pckfl=open(pckloc, 'wb'); pickle.dump(inactivecells, pckfl); pckfl.close()
            #del inactivecells; inactivecells={}       
    ###create a celldictionary populated with cell_class
    celldct={}
    for keys,values in inactivecells.iteritems():
        tmp=cell_class(keys); [tmp.add_connection(xx) for xx in values]
        celldct[keys]=tmp
    ###add z plns that each cell spans; NOTE THE PLANES ARE IN XY FROM CV2
    for keys, values in celldct.iteritems():
        values.add_pln(cell_lst_srtd[keys])        
        [values.add_pln(cell_lst_srtd[xx]) for xx in values.connectedwith]        
    ###find centers
    tplst=find_radius_and_center(celldct) ###numbers are in PIXELS
    ########remove any cells that have been detected from plane before validzplns, meaning they should have been picked up by the previous array job    
    prunedtplst=[]    
    for i in tplst:
        if (valid_plns[0] - 1 ) not in [xx[0] for xx in i.plns.values()]:
            prunedtplst.append(i)
    tplst=prunedtplst; del prunedtplst
    ##### split cells only detected on a single plane. An extra filtering step, should be turned off for large z-steps, especially with small sheet NA's
    tplstm=[]; tplsts=[]
    for cells in tplst:
        if len(cells.connectedwith) > 1:
            tplstm.append(cells)
        else:
            tplsts.append(cells)
    del tplst
    ###check for errors in having active cells still in last overlap plane   
    if len(activecells) > 0:
        writer(outdr, 'active cell final list was {}\n this means that cells extended PAST overlap area, increase pln_chnk until this value is zero'.format(len(activecells)), flnm='celldetect3d_ERROR.txt')          
    ###save centers to .np file; can adjust to save other components
    print ('saving center file as ch{}_cells_for_plns{}-{}.p'.format(ch.channel, valid_plns[0], valid_plns[-1]))            
    tmpdct={'multi': tplstm, 'single': tplsts}    
    pckloc=os.path.join(celldetect3dfld, 'ch{}_cells_for_plns{}-{}.p'.format(ch.channel, valid_plns[0], valid_plns[-1])); pckfl=open(pckloc, 'wb'); pickle.dump(tmpdct, pckfl); pckfl.close()     
    print ('time taken for cell detection of ch{}, {} zplns was {} minutes'.format(ch.channel, len(valid_plns), (time.time()-bigstart)/60 ))
    writer(outdr, '\nSTEP 4:\n   Jobid: {}\n    ch{}_cells_for_plns{}-{}.p in {} minutes'.format(jobid, ch.channel, valid_plns[0], valid_plns[-1], (time.time()-bigstart)/60), flnm='step4_out.txt')    
    gc.collect(); del zdct, cell_lst
    p.terminate()
    return

#%%
def cell_connected_check((mxdst, cells, activecells, inactivecells, cells_in_pln, cell_lst_srtd, core, cores)):
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
#%%
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
    im1=np.zeros((ymax-ymin, xmax-xmin)); cv2.fillPoly(im1, [j-[xmin, ymin]], color=(255,255, 255));  #and here flipping of yx on the cv2 part
    im2=np.zeros((ymax-ymin, xmax-xmin)); cv2.fillPoly(im2, [ii-[xmin, ymin]], color=(255,255, 255)); #and here flipping of yx on the cv2 part
    if np.argmax(im1*im2)>0:
        del im1, im2, cell_lst_srtd, ymin, xmin, ymax, xmax 
        gc.collect()
        return 1
    else:
        del im1, im2, cell_lst_srtd, ymin, xmin, ymax, xmax
        gc.collect()
        return 0    
    
#%%
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
#%%

#%%
###################################################################################################################################################
###################################################################################################################################################
###################################################################STEP 5##########################################################################
###################################################################################################################################################
###################################################################################################################################################


#%%
###for a cell, apply a color variable, ungroup cells, sort by z. Array job to mark cells and then save
def cell_ovly_3d((svlc, tplst, valid_plns, outdr, vol_to_process, resizefactor, cell_class_lst, multiplane, ovly, core, cores)):
    '''function to apply independent colors to cells that have been detected on multiple planes, and save images. 
    This has been parallelized and calculates a range based on the number of cores.
    '''
    ####assign random color to each group of cells, the break up groups, and sort by plane    
    #use this for looking at multi cells only
    if multiplane != False:       
        multilist=[x for x in tplst if len(x.plns) >1]
        tplst=multilist
    cell_color_lst=[]        
    for cells in tplst:            
        color=(random.randint(50,255), random.randint(50,255), random.randint(50,255))
        for plns in cells.plns.itervalues():
            cell_color_lst.append([plns, color]) ###break apart list        
    #sort by z
    try: #sort by z
        cell_color_lst.sort()
    except ValueError: #done to suppress error
        pass
    ############################################load find full size files############################################
    makedir(svlc)
    dr=vol_to_process.full_sizedatafld_vol
    #dr='/home/wanglab/wang/pisano/tracing_output/H129_EGFP_NLS_vc22_01_lob5_intensityrescale/full_sizedatafld/ch01'
    fl=[f for f in os.listdir(dr) if '.tif' in f]
    reg=re.compile(r'(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.tif)') ###regex used since code changes file format
    matches=map(reg.match, fl)
    ##find index of z,y,x,ch in matches
    z_indx=matches[0].span('z')
    ####make dictionary where each pln is a key, values are all point that should be drawn
    zdct={}
    for i in range(len(cell_color_lst)):
        pln=cell_color_lst[i] ##check
        try:        
            zdct[str(pln[0][0]).zfill(4)].append(pln)
        except KeyError:
            zdct[str(pln[0][0]).zfill(4)]=[]
            zdct[str(pln[0][0]).zfill(4)].append(pln)
    ############parse jobs:
    chnkrng=chunkit(core, cores, zdct)
    ########### apply points to each pln
    if ovly == True:    ###ovly 3d cells onto data
        flpth=''.join(matches[0].groups())
        y,x=tifffile.imread(os.path.join(dr, flpth)).shape
        dsf=resizefactor
        for i in range(chnkrng[0], chnkrng[1]):
            try:        
                pln=zdct.keys()[i]; cntrs=zdct[zdct.keys()[i]]
                ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs)
                del cntrs, pln; gc.collect()
            except:
                pass
        ###make zpln range and make empty planes on the zplns where no cells are found    
        try:
            plnlst=[int(plns) for plns in zdct.iterkeys()]; plnrng=range(min(plnlst), max(plnlst)+1)
            nocell_plns=set(plnrng).difference(set(plnlst))
                ############parse jobs:  
            chnkrng=chunkit(core, cores, nocell_plns)
            print ('\n\ncells not found on planes: {}\nmaking empty images for those planes...'.format(nocell_plns))
            for i in range(chnkrng[0], chnkrng[1]):
                pln=list(nocell_plns)[i]
                blnk_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln)
                del pln; gc.collect()
            return 
        except ValueError: #no cells found
            print ('No Cells found, not making color stack...')
            return 
    elif ovly == False:  ###make downsized 3d cells
        flpth=''.join(matches[0].groups())
        y,x=tifffile.imread(os.path.join(dr, flpth)).shape
        dsf=resizefactor
        for i in range(chnkrng[0], chnkrng[1]):
            try:        
                pln=zdct.keys()[i]; cntrs=zdct[zdct.keys()[i]]
                no_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs)
                del cntrs, pln; gc.collect()
            except:
                pass
        ###make zpln range and make empty planes on the zplns where no cells are found    
        try:
            plnlst=[int(plns) for plns in zdct.iterkeys()]; plnrng=range(min(plnlst), max(plnlst)+1)
            nocell_plns=set(plnrng).difference(set(plnlst))
            if len(nocell_plns) != 0:
                print ('\n\ncells not found on planes: {}\nmaking empty images for those planes...'.format(nocell_plns))
                ############parse jobs:  
            chnkrng=chunkit(core, cores, nocell_plns)
            for i in range(chnkrng[0], chnkrng[1]):
                try:                
                    pln=list(nocell_plns)[i]
                    blnk_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln)
                    del pln; gc.collect()
                except IndexError:
                    pass
            return 
        except ValueError: #no cells found
            print ('No Cells found, not making color stack...')
            return 
#%%
def ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=cv2.imread(os.path.join(dr, flpth1), 1)
    for cntr in cntrs:            
        cv2.fillConvexPoly(im, cntr[0][3], color=cntr[1])
    tifffile.imsave(os.path.join(svlc, '3DCELLS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('processed pln {}, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs, im
    gc.collect()
    return
def blnk_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln):
    flpth1=flpth[:z_indx[0]]+str(pln).zfill(4)+flpth[z_indx[1]:]
    im=cv2.imread(os.path.join(dr, flpth1), 1)
    tifffile.imsave(os.path.join(svlc, '3DCELLS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('No cells found on pln {}, making empty file, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, im
    gc.collect()
    return
def no_ovly_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=np.zeros((y,x))
    for cntr in cntrs:            
        cv2.fillConvexPoly(im, cntr[0][3], color=cntr[1])
    tifffile.imsave(os.path.join(svlc, '3DCELLS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('processed pln {}, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, cntrs, im
    gc.collect()
    return
def blnk_helper(flpth, z_indx, dr, svlc, x, y, dsf, pln):
    flpth1=flpth[:z_indx[0]]+str(pln)+flpth[z_indx[1]:]    
    im=np.zeros((y,x))
    tifffile.imsave(os.path.join(svlc, '3DCELLS_'+flpth1), cv2.resize(im, (int(x/dsf), int(y/dsf)), interpolation=cv2.INTER_AREA).astype('uint8'))
    #print ('No cells found on pln {}, making empty file, saved in:\n{}'.format(pln, svlc))
    del flpth1, flpth, z_indx, dr, svlc, x, y, dsf, pln, im
    gc.collect()
    return


#%%
def identify_structures_w_cells(jobid, cores=5, make_color_images=False, overlay_on_original_data=False, consider_only_multipln_cells=False, **kwargs):
    '''function to take 3d detected cells and apply elastix transform
    '''    
    #######################inputs and setup#################################################
    ###inputs    
    outdr=kwargs['outputdirectory']
    pckl=open(os.path.join(outdr, 'param_dict.p'), 'rb'); kwargs.update(pickle.load(pckl)); pckl.close()    
    vols=kwargs['volumes']
    reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]
    celldetect3dfld = reg_vol.celldetect3dfld
    xscl, yscl, zscl = reg_vol.xyz_scale ###micron/pixel  
    cellcoordinatesfld = reg_vol.cellcoordinatesfld
    zmx, ymx, xmx = reg_vol.fullsizedimensions
    AtlasFile=reg_vol.atlasfile    
    print ('Using {} CORES'.format(cores))
    try:
        p
    except NameError:
        p=mp.Pool(cores)
    full_sizedatafld=reg_vol.full_sizedatafld
    resizefactor=kwargs['resizefactor']
    brainname=reg_vol.brainname
    ############################################################################################################
    #######################use regex to sort np files by ch and then by zpln####################################
    ############################################################################################################    
    fl=[f for f in os.listdir(celldetect3dfld) if '.p' in f and 'ch' in f] #sorted for raw files
    reg=re.compile(r'(.*h+)(?P<ch>\d{2})(.*)(.p)')
    matches=map(reg.match, fl)      
    ###determine cell volumes and set job
    cell_vols=[xx for xx in vols if xx.ch_type == 'cellch']; cell_vols.sort()
    vol_to_process=cell_vols[jobid] ###each job represents a different cell volume
    ch=vol_to_process.channel
    ###get rid of extra jobs
    if jobid >= len(cell_vols): ###used to end jobs if too many are called
        print ('jobid({}) >= cell volumes {}'.format(jobid, len(cell_vols)))         
        return
    ##load .np files
    sys.stdout.write('jobid({}), loading ch{} .p files to extract cell_class objects....'.format(jobid, ch))
    cell_class_lst=[]
    for fl in [os.path.join(celldetect3dfld, ''.join(xx.groups())) for xx in matches if xx.group('ch')[-2:] in ch]:
        tmpkwargs={}
        pckl=open(fl, 'rb'); tmpkwargs.update(pickle.load(pckl)); pckl.close()
        if consider_only_multipln_cells == False:
            tmplst=tmpkwargs['single']
            [tmplst.append(xx) for xx in tmpkwargs['multi']]
        elif consider_only_multipln_cells == True:    
            tmplst=tmpkwargs['multi']
        [cell_class_lst.append(xx) for xx in tmplst]
    sys.stdout.write('done loading cell_class objects.\n')  
    if len(cell_class_lst) == 0:
        print ('Length of cells in ch{} was zero, ending process...'.format(len(cell_class_lst)))        
        return
    ############################################################################################################        
    ##############################make color files##############################################################
    ############################################################################################################    
    if make_color_images == True:
        sys.stdout.write('making 3d planes...')
        sys.stdout.flush()
        valid_plns=range(0, zmx+1)        
        svlc=os.path.join(outdr, 'ch{}_3dcells'.format(ch)); removedir(svlc)       
        if overlay_on_original_data == False:
            ovly = False
        elif overlay_on_original_data == True:
            ovly = True
        iterlst=[]; [iterlst.append((svlc, cell_class_lst, valid_plns, outdr, vol_to_process, resizefactor, cell_class_lst, consider_only_multipln_cells, ovly, core, cores)) for core in range(cores)]
        p.map(cell_ovly_3d, iterlst); 
        lst=os.listdir(svlc); lst1=[os.path.join(svlc, xx) for xx in lst]; lst1.sort(); del lst; del iterlst
        ###load ims and return dct of keys=str(zpln), values=np.array          
        sys.stdout.write('3d planes made, saved in {},\nnow compressing into single tifffile'.format(svlc))        
        imstack=tifffile.imread(lst1); del lst1
        if len(imstack.shape) > 3:    
            imstack=np.squeeze(imstack)    
        
        ###account for orientation differences, i.e. from horiztonal scan to sagittal for atlas registration       
        imstack = fix_orientation(imstack, **kwargs)

        tiffstackpth=os.path.join(outdr, '3D_cells_ch{}_{}'.format(ch, brainname))
        tifffile.imsave(tiffstackpth,imstack.astype('uint16')); del imstack; gc.collect()
        shutil.rmtree(svlc)
        sys.stdout.write('color image stack made for ch{}'.format(ch))
    else:
        sys.stdout.write('make_color_images=False, not creating images')        
    ############################################################################################################        
    ######################apply point transform and make transformix input file#################################
    ############################################################################################################   
    ###find centers and add 1's to make nx4 array for affine matrix multiplication to account for downsizing
    ###everything is in PIXELS
    cellarr=np.empty((len(cell_class_lst),3));
    for i in range(len(cell_class_lst)):
        cellarr[i,...]=cell_class_lst[i].center ###full sized dimensions: if 3x3 tiles z(~2000),y(7680),x(6480) before any rotation 
    
    #account for orientation changes in cell array and dimensions
    cellarr = fix_contour_orientation(cellarr, **kwargs)
    z,y,x = fix_dimension_orientation(**kwargs)
        
    '''#old before fix_orientaiton functions    
    try:        
        cellarr=swap_cols(cellarr, *kwargs['swapaxes']) ###change columns to account for orientation changes between brain and atlas: if horizontal to sagittal==>x,y,z relative to horizontal; zyx relative to sagittal
        z,y,x=swap_cols(np.array([vol_to_process.fullsizedimensions]), *kwargs['swapaxes'])[0]##convert full size cooridnates into sagittal atlas coordinates
        sys.stdout.write('Swapping Axes')
    except: ###if no swapaxes then just take normal z,y,x dimensions in original scan orientation
        z,y,x=vol_to_process.fullsizedimensions
        sys.stdout.write('No Swapping of Axes')
        
    '''
    d1,d2=cellarr.shape
    nx4centers=np.ones((d1,d2+1))
    nx4centers[:,:-1]=cellarr
    ###find resampled elastix file dim
    with tifffile.TiffFile([os.path.join(outdr, f) for f in os.listdir(outdr) if 'resampledforelastix' in f and 'ch{}'.format([xx.channel for xx in vols if xx.ch_type == 'regch'][0]) in f][0]) as tif:  
        zr=len(tif.pages)
        yr,xr=tif.pages[0].shape
        tif.close()
    ####create transformmatrix
    trnsfrmmatrix=np.identity(4)*(zr/z, yr/y, xr/x, 1) ###downscale to "resampledforelastix size"
    sys.stdout.write('trnsfrmmatrix:\n{}\n'.format(trnsfrmmatrix))
    #nx4 * 4x4 to give transform
    trnsfmdpnts=nx4centers.dot(trnsfrmmatrix) ##z,y,x
    sys.stdout.write('first three transformed pnts:\n{}\n'.format(trnsfmdpnts[0:3]))
    #create txt file, with elastix header, then populate points
    txtflnm='{}_zyx_transformedpnts_ch{}.txt'.format(brainname, ch)
    pnts_fld=os.path.join(outdr, 'transformedpoints_pretransformix'); makedir(pnts_fld)
    transforminput=os.path.join(pnts_fld, txtflnm)
    removedir(transforminput)###prevent adding to an already made file
    writer(pnts_fld, 'index\n{}\n'.format(len(trnsfmdpnts)), flnm=txtflnm)    
    sys.stdout.write('writing centers to transfomix input points text file: {}....'.format(transforminput))
    [writer(pnts_fld, '{} {} {}\n'.format(i[2],i[1],i[0]), flnm=txtflnm, verbose=False) for i in trnsfmdpnts] ####this step converts from zyx to xyz*****
    sys.stdout.write('done writing centers.'); sys.stdout.flush()
    del trnsfmdpnts, trnsfrmmatrix, nx4centers, cellarr; gc.collect()
    ############################################################################################################        
    ####################################elastix for inverse transform###########################################
    ############################################################################################################     
    transformfile=make_inverse_transform(**kwargs)
    assert os.path.exists(transformfile)
    sys.stdout.write('***Elastix Inverse Transform File: {}***'.format(transformfile))
    ############################################################################################################        
    ####################################transformix#############################################################
    ############################################################################################################        
    if make_color_images != False:
        #apply transform to 3d_tiffstack:
        transformimageinput=tiffstackpth; elastixpth=os.path.join(outdr, 'elastix')
        trnsfrm_outpath=os.path.join(elastixpth, '3D_cells_ch{}_{}'.format(ch, brainname)); makedir(trnsfrm_outpath)
        writer(trnsfrm_outpath,'\nProcessing ch{} 3D...'.format(ch))
        #transformfiles=[os.path.join(elastixpth, xx) for xx in os.listdir(os.path.join(outdr, 'elastix')) if "TransformParameters" in xx]; mxx=max([xx[-5] for xx in transformfiles])
        #transformfile=os.path.join(elastixpth, 'TransformParameters.{}.txt'.format(mxx))
        trnsfrm_out_file = os.path.join(trnsfrm_outpath, 'result.tif') #output of transformix
        transformimageinput_resized=transformimageinput[:-4]+'_resampledforelastix.tif'
        print ('Resizing {}'.format(transformimageinput_resized))        
        resample_par(cores, transformimageinput, AtlasFile, svlocname=transformimageinput_resized, singletifffile=True, resamplefactor=1.7)                
        sp.call(['transformix', '-in', transformimageinput_resized, '-out', trnsfrm_outpath, '-tp', transformfile])
        writer(trnsfrm_outpath,'\n   Transformix File Generated: {}'.format(trnsfrm_out_file))
        writer(trnsfrm_outpath,'\n   Passing colorcode: {} file as {}'.format(ch, os.path.join(trnsfrm_outpath, 'depthcoded.png')))
        ###depth coded image of transformix result; not functional yet
        #depth.colorcode(trnsfrm_out_file, trnsfrm_outpath)
        #getvoxels(trnsfrm_out_file, os.path.join(trnsfrm_outpath, 'zyx_voxels_{}.npy'.format(ch)))
        #allen_compare(AtlasFile, svlc, trnsfrm_outpath)
        ##if successful delete cell cooridnates and maybe celldetect3d flds
    ############################################################    
    ##############apply transform to points#####################
    elastixpth=os.path.join(outdr, 'elastix_inverse_transform')
    trnsfrm_outpath=os.path.join(elastixpth, 'ch{}_3dpoints'.format(ch)); makedir(trnsfrm_outpath)
    writer(trnsfrm_outpath,'\n***********Starting Transformix for: {}***********'.format(ch)); sys.stdout.flush()
    #transformfiles=[os.path.join(elastixpth, xx) for xx in os.listdir(os.path.join(outdr, 'elastix')) if "TransformParameters" in xx]; mxx=max([xx[-5] for xx in transformfiles])
    #transformfile=os.path.join(elastixpth, 'TransformParameters.{}.txt'.format(mxx))
    trnsfrm_out_file = os.path.join(trnsfrm_outpath, 'outputpoints.txt') #MIGHT NEED TO CHANGE THIS
    sp.call(['transformix', '-def', transforminput, '-out', trnsfrm_outpath, '-tp', transformfile])
    #sp.call(['transformix', '-def', 'all', '-out', trnsfrm_outpath, '-tp', transformfile]) ##displacement field
    writer(trnsfrm_outpath,'\n   Transformix File Generated: {}'.format(trnsfrm_out_file))
    ####################################################################################    
    ##############generate list and image overlaid onto allen atlas#####################
    ####################################################################################    
    transformed_pnts_to_allen(trnsfrm_out_file, ch, cores, **kwargs)
    try:
        p.terminate()
    except:
        1
    print ('end of script')
    return

#%%
    
def transformed_pnts_to_allen(trnsfrm_out_file, ch, cores, point_or_index=None, **kwargs):
    '''function to take elastix point transform file and return anatomical locations of those points
    point_or_index=None/point/index: determines which transformix output to use: point is more accurate, index is pixel value(?)
    Elastix uses the xyz convention rather than the zyx numpy convention
    '''    
    #####inputs 
    assert type(trnsfrm_out_file)==str
    if point_or_index==None:
        point_or_index = 'OutputPoint'
    elif point_or_index == 'point':
        point_or_index = 'OutputPoint'
    elif point_or_index == 'index':
        point_or_index = 'OutputIndexFixed'
    try: #check to see if pool processes have already been spawned
        p
    except NameError:
        p=mp.Pool(cores)
    vols=kwargs['volumes']
    reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]
    #####testing
    #allen_id_table='/home/wanglab/temp_wang/pisano/Python/lightsheet/supp_files/allen_id_table.xlsx'
    #annotationfile='/home/wanglab/temp_wang/pisano/tracing_output/allenatlas/annotation_25_ccf2015.nrrd'
    #trnsfrm_out_file='/home/wanglab/temp_wang/pisano/tracing_output/092915_crsi_02/bl6_092915_crsi_02_w488_z3um_100msec_na0005_sw3862_1hfds_zyx_transformedpnts_ch02.txt'
    ####load files
    allen_id_table=pd.read_excel(os.path.join(reg_vol.packagedirectory, 'supp_files/allen_id_table.xlsx')) ##use for determining neuroanatomical locations according to allen
    ann=sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile'])) ###zyx
    with open(trnsfrm_out_file, "rb") as f:                
        lines=f.readlines()
        f.close()
    #####populate post-transformed array of cell centers
    sys.stdout.write('{} points detected'.format(len(lines)))
    arr=np.empty((len(lines), 3))    
    for i in range(len(lines)):        
        arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
    #iterlst=[]; [iterlst.append((arr, ann, core, cores)) for core in range(cores)]
    #lst=p.map(transformed_pnts_to_allen_helper_func_par, iterlst); del iterlst
    #pnt_lst=[xx for x in lst for xx in x]        
    pnts=transformed_pnts_to_allen_helper_func(arr, ann); pnt_lst=[xx for xx in pnts if xx != 0]
    if len(pnt_lst)==0:
        raise ValueError('pnt_lst is empty')
    else:
        sys.stdout.write('len of pnt_lst({})'.format(len(pnt_lst)))
    imstack = brain_structure_keeper(ann, True, *pnt_lst) ###annotation file, true=to depict density, list of pnts
    df=structure_lister(allen_id_table, *pnt_lst)
    #########save out imstack and df
    tifffile.imsave(os.path.join(kwargs['outputdirectory'], 'ch{}_{}_structure_density_map.tif'.format(ch, reg_vol.brainname)), imstack)
    excelfl=os.path.join(kwargs['outputdirectory'], 'ch{}_{}_stuctures_table.xlsx'.format(ch, reg_vol.brainname))    
    df.to_excel(excelfl)
    print ('file saved as: {}'.format(excelfl))
    try:
        p.terminate()
    except:
        1
    return
    
def transformed_pnts_to_allen_helper_func_par((arr, ann, core, cores)):
    ############parse jobs:
    chnkrng=chunkit(core, cores, arr)
    ########procecss
    pnt_lst=[]; tick=0
    for i in range(chnkrng[0], chnkrng[1]):
        try:        
            pnt=[int(x) for x in arr[i]]
            pnt_lst.append(ann[pnt[2], pnt[1], pnt[0]]) ###find pixel id; arr=XYZ; ann=ZYX
        except IndexError:
            tick+=1
            pass ######THIS NEEDS TO BE CHECKED BUT I BELIEVE INDEXES WILL BE OUT OF 
    print tick
    return pnt_lst
def transformed_pnts_to_allen_helper_func(arr, ann):
    ########procecss
    pnt_lst=[]; tick=0
    for i in range(len(arr)):
        try:        
            pnt=[int(x) for x in arr[i]]
            pnt_lst.append(ann[pnt[2], pnt[1], pnt[0]]) ###find pixel id; arr=XYZ; ann=ZYX
        except IndexError:
            tick+=1
            pass ######THIS NEEDS TO BE CHECKED BUT I BELIEVE INDEXES WILL BE OUT OF 
    print tick
    return pnt_lst
        
def structure_lister(allen_id_table, *args):
    '''Function that generates a pd table of structures where cell detection has been observed
    Inputs:
        allen_id_table=annotation file as np.array
        *args=list of allen ID pixel values
    '''
    df=allen_id_table.assign(cell_count= [0]*len(allen_id_table)) #add count columns to df
    #regions_w_cells=[]    
    for i in args:
        df.ix[df.id==i, 'cell_count']=df.ix[df.id==i, 'cell_count']+1 #increase the cell count by 1
    #regions_w_cells.append(df[df.cell_count>0])
    #return regions_w_cells
    return df
        
#%%    
def brain_structure_keeper(ann, depictdensity, *args):
    '''Function that generates an image of structures where cell detection has been observed
    Inputs:
        ann=annotation file as np.array
        depictdensity=True/False, if true normalizes cell counts per/region and normalizes them to upper half of 8bit pixel values
        *args=list of allen ID pixel values
    ''' 
    ###############################UNFINISHED    
    ##find zyx coordinates of args
    if depictdensity==True:        
        dct=Counter(args) #keys=number, #values=count
        mostcellsinloc=max(dct.values())
        leastcellsinloc=min(dct.values())
        #zip(np.linspace(leastcellsinloc, mostcellsinloc), np.logspace(127, 255))
        tick=0
        #pxrng=np.linspace(127, 255, int(len(dct)/2))       
        stack=np.zeros(ann.shape).astype('uint8') ##65000
        #b=[((count - np.mean(dct.values())) / np.std(dct.values())) for count in dct.itervalues()]
        for pixid, count in dct.iteritems():
            pixintensity= ((count - leastcellsinloc ) / (mostcellsinloc - leastcellsinloc) * 255) #+ 127.5 ###scaling to upper half of pixel values; done to prevent zeroing out of minimum
            #pixintensity= (count - np.mean(dct.values())) / np.std(dct.values()); print pixintensity
            stack[ann==pixid] = pixintensity
            tick+=1
            print('done {} in {}'.format(tick, len(dct)))
        return stack
        ##################WORKING
    elif depictdensity == False:
        argss=list(set(args))
        stack=np.zeros(ann.shape).astype('uint8')
        for i in argss:
            stack[ann==i] = 255
        return stack

#%%       
def make_inverse_transform(**kwargs):
    '''Script to perform inverse transform and return path to elastix inverse parameter file
    '''
    ############inputs        
    outdr = kwargs['outputdirectory']
    pckl = open(os.path.join(outdr, 'param_dict.p'), 'rb'); kwargs.update(pickle.load(pckl)); pckl.close()    
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]
    xscl, yscl, zscl = reg_vol.xyz_scale ###micron/pixel  
    zmx, ymx, xmx = reg_vol.fullsizedimensions
    AtlasFile = reg_vol.atlasfile    
    #resizedregdct = kwargs['resizedregchtif'] 
    parameterfolder = reg_vol.parameterfolder
    
    ###############
    ###images need to have been stitched, resized, and saved into single tiff stack ###
    ###resize to ~220% total size of atlas (1.3x/dim) ###    
    reg_vol.add_resampled_for_elastix_vol(reg_vol.downsized_vol+'_resampledforelastix.tif')
    #resample_par(cores, reg_vol, AtlasFile, svlocname=reg_vol_resampled, singletifffile=True, resamplefactor=1.2)
    if os.path.exists(reg_vol.resampled_for_elastix_vol) == False:
        print ('Resizing')        
        #resample(reg_vol, AtlasFile, svlocname=reg_vol_resampled, singletifffile=True, resamplefactor=1.3)
        resample_par(cores, reg_vol.downsized_vol+'.tif', AtlasFile, svlocname=reg_vol.resampled_for_elastix_vol, singletifffile=True, resamplefactor=1.3)
        print ('Past Resizing')
    ####setup
    parameters = []; [parameters.append(files) for files in os.listdir(parameterfolder) if files[0] != '.' and files [-1] != '~']; parameters.sort()       
    ###set up save locations
    svlc = os.path.join(outdr, 'elastix_inverse_transform'); makedir(svlc)
    ###Creating LogFile
    writer(svlc, 'Starting elastix...AtlasFile: {}\n   parameterfolder: {}\n   svlc: {}\n'.format(AtlasFile, parameterfolder, svlc))
    writer(svlc, 'Order of parameters used in Elastix:{}\n...\n\n'.format(parameters))
    ### setup elastix command and running using parameters 
    print ('***Running Elastix***')         
    e_params=['elastix', '-f', reg_vol.resampled_for_elastix_vol, '-m', AtlasFile, '-out', svlc]
    ###adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(os.path.join(parameterfolder, parameters[x]))
    writer(svlc,'Elastix Command:\n{}\n...'.format(e_params))
    ####################################################run elastix#####################################################################
    e_transform_file = os.path.join(svlc, 'TransformParameters.{}.txt'.format(len([x for x in os.listdir(parameterfolder) if '~' not in x])-1))
    if os.path.exists(e_transform_file) == False:
        sp.call(e_params)
    assert os.path.exists(e_transform_file)
    writer(svlc,'***Elastix Registration Successfully Completed***\n')
    writer(svlc,'\ne_transform_file is {}'.format(e_transform_file))    
    ####################
    return e_transform_file     
        
#%%
def detect_cells_in_3d_checker(pln_chnk=50, **kwargs):
    outdr = kwargs['outputdirectory']
    pckl = open(os.path.join(outdr, 'param_dict.p'), 'rb'); kwargs.update(pickle.load(pckl)); pckl.close()    
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]
    celldetect3dfld = vols[0].celldetect3dfld
    zmax = vols[0].fullsizedimensions[0]
    if len(os.listdir(celldetect3dfld)) != int(ceil(zmax / pln_chnk)) + 1:
        writer(outdr, '\n\n***************************STEP 4.1 FAILED*********************\n{} files found in {}. Should have {}.'.format(len(os.listdir(celldetect3dfld)), celldetect3dfld[celldetect3dfld.rfind('/')+1:], int(ceil(zmax / pln_chnk))+1))
    else:
        writer(outdr, '\n\n***************************STEP 4.1 SUCCESS*********************\n{} files found in {}. Should have {}.'.format(len(os.listdir(celldetect3dfld)), celldetect3dfld[celldetect3dfld.rfind('/')+1:], int(ceil(zmax / pln_chnk))+1))
    return
        
