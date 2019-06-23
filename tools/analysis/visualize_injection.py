# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:50:35 2016

@author: wanglab
"""
from __future__ import division
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs
from tools.utils.directorydeterminer import directorydeterminer
from tools.utils.update import find_all_brains
import pandas as pd, os, numpy as np, cv2, shutil, sys
from tools.imageprocessing.preprocessing import resample_par
from tools.registration.register import allen_compare
from skimage.external import tifffile
import SimpleITK as sitk
from tools.objectdetection.find_injection import detect_inj_site
from tools.registration.allen_structure_json_to_pandas import isolate_and_overlay, isolate_structures
from tools.registration.transform import transformed_pnts_to_allen_helper_func, swap_cols, filter_overinterpolated_pixels
from tools.imageprocessing import depth
import cPickle as pickle
import matplotlib.pyplot as pl 
from matplotlib.colors import colorConverter
from tools.analysis.network_analysis import generate_multi_radial_plots_using_lists
from scipy.ndimage.interpolation import zoom



#%%
if __name__ == '__main__':
    
    ##detect inj site still seems broken    
    
    #Single Brain 
    svlc='/home/wanglab/Desktop/depthoverlay/single'
    pth = '/home/wanglab/wang/pisano/tracing_output/bl6_crII/db_bl6_20160622_crII_52-5hr_01_badper'
    generate_inj_fig(pth, svlc, bglst=False, cerebellum=True, threshold = 10000, detectinjsite=True)
    generate_inj_fig(pth, svlc, bglst=False, cerebellum=True, threshold = 10000, detectinjsite=False, cleanup=False)
    
    #multibrains:
    svlc='/home/wanglab/Desktop/depthoverlay/outline'
    pthlst = ['/home/wanglab/wang/pisano/tracing_output/bl6_crII/db_bl6_20160622_crII_52-5hr_01_badper',  '/home/wanglab/wang/pisano/tracing_output/vc22/vc22_02_lob6']
    generate_inj_fig_multi_old(pthlst, svlc, bglst=False, cerebellum=True, threshold = 10000, detectinjsite=False)    
    
    #all brains:
    #set pth
    tracing_output_fld = '/home/wanglab/wang/pisano/tracing_output'
    
    #just find all brains:
    svlc='/home/wanglab/Desktop/depthoverlay/L7CreTS'    
    ann_pth = '/home/wanglab/wang/pisano/Python/allenatlas/annotation_25_ccf2015_reflectedhorizontally.nrrd'
    allbrainpths = find_all_brains(tracing_output_fld)
    nonprv = [xx for xx in allbrainpths if 'prv' not in xx]; 
    nonprv.remove('/home/wanglab/wang/pisano/tracing_output/bl6_crII/20160628_bl6_crii_200r_01')
    nonprv.remove('/home/wanglab/wang/pisano/tracing_output/sd_hsv_lob6/sd_hsv_ml0_500down')
    l7s = [xx for xx in nonprv if 'l7cre_ts' in xx]; l7s.sort()
    
    ###setup:
    names = ['L7-Cre 42 hrs', 'L7-Cre 50 hrs', 'L7-Cre 38 hrs', 'L7-Cre 57 hrs', 'L7-Cre 70 hrs', 'L7-Cre 58 hrs', 'L7-Cre 64 hrs']
    
    l7dct = dict(zip(names, l7s))

    #tmp    
    del l7dct['L7-Cre 57 hrs']
    del l7dct['L7-Cre 42 hrs']
    del l7dct['L7-Cre 64 hrs']
    del l7dct['L7-Cre 38 hrs']
    del l7dct['L7-Cre 50 hrs']

    #make functionality so that it accepts the same dictionary that generate_multi_radial_plots_using_lists takes
    generate_inj_fig_multi_old_old(l7dct, svlc, bglst=False, cerebellum=True, threshold = 300, detectinjsite=False)    

    ##finish this part 
    generate_multi_radial_plots_using_lists(l7dct, title_substructure_list = False, title = 'L7TimeSeries', nametype='acronym', svlocname = svlc + '/multiradialplot_acronyms.png', remove_childless_structures_not_repsented_in_ABA = True, ann_pth=ann_pth)
    
#%%
def generate_inj_fig_multi_alpha(pths, svlc, bglst=False, crop=False, injsitethreshold = False, colorlst = False, cleanup=True, alpha = 0.95, bilateraloutlines=True, testing = False):    
    '''Function to generate outlines of structres of interest
    NOTE THIS REQUIRES THAT "STEP3" elastix registration to be run or tools.utils.update_inj_sites
    Inputs
    -------------
    pthlst = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = savelocation
    bglst = structures to outline in the following format: [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726]]    
    crop = 
          cerebellum: '[:,390:,:]'
          caudal midbrain: '[:,300:415,:]'
          midbrain: '[:,215:415,:]'
          thalamus: '[:,215:345,:]'
          anterior cortex: '[:,:250,:]'
    injsitethreshold = 
                int  --> use this function to detect injection site via median blurring and thresholding (value of this parameter)
                false --> use contours found by light sheet package (from tools.utils.update_inj_sites)
    colorlst = ['darksalmon', 'lightgreen', 'skyblue', 'b', 'r', 'g', 'm', 'y', 'c', 'darkturquoise', 'lime', 'firebrick', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']
    cleanup = True to delete intermediate files that are generated
    bilateraloutlines=True; if false will only display single sided outlines
    testing = False used to make sure orientation is correct - THIS WILL STOP THE DISPLAY OF ACTUAL INJECTION SITES, literally just for testing
    
    alpha = transparency level for filled injection contours
    
    '''
    makedir(svlc)
    rmlst = []
    cntdct = {}
    
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct
    
    #load kwargs
    for name, pth in pths.iteritems():
        ##inputs:    
        kwargs={}  
        try:        
            with open(os.path.join(pth, 'param_dict.p'), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
        except IOError:
            with open(os.path.join(pth, 'param_dict.p'), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
                
        if directorydeterminer() == '/home/wanglab/': kwargs.update(pth_update(kwargs))
        
        svlc_brain=os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(svlc_brain); rmlst.append(svlc_brain)
        
        #set up allen/ann stuff
        sys.stdout.write('\n\nStarting {}...'.format(name)); sys.stdout.flush()
        annfl = pth_update(kwargs['annotationfile'])        
        sys.stdout.write('\n     Annotation File: {}'.format(annfl[annfl.rfind('/')+1:])); sys.stdout.flush()        
        ann = sitk.ReadImage(pth_update(kwargs['annotationfile']))

        #bglst        
        if not bglst: bglst = [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726], ['CrI_II', 1017, 1056, 10677,10676,10675, 1064,10680,10679,10678]]
                    
        #if first
        if pth == pths.values()[0]:        
            #make outline for each structure in bglst
            sys.stdout.write('\n     generating outlines for bglst structures (time consuming, only done once)...'); sys.stdout.flush()            
            cntlsts = []
            for lst in bglst:
                nm = lst[0]        
                if crop != False: 
                    im = eval("isolate_structures(pth_update(kwargs['annotationfile']), *lst[1:]){}".format(crop))
                    ann = eval("sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))".format(crop))
                else:
                    im = isolate_structures(pth_update(kwargs['annotationfile']), *lst[1:])
                    ann = sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))
                    
        
                
                svlc2=os.path.join(svlc_brain, nm)
                makedir(svlc2); rmlst.append(svlc2)
                impth=os.path.join(svlc2, str(pth_update(kwargs['volumes'][0].brainname)+'.tif'))    
                tifffile.imsave(impth, im.astype('uint8'))
                
                allen = os.path.join(svlc_brain, 'allenatlas'); makedir(allen); rmlst.append(allen)
                if crop != False: 
                    croppedAtlasFile = os.path.join(allen, 'cropped_atlas.tif')   
                    eval("tifffile.imsave(croppedAtlasFile, tifffile.imread(pth_update(kwargs['AtlasFile'])){})".format(crop))
                    cnts = allen_compare(croppedAtlasFile, svlc_brain, impth, outline = True)
                else:
                    cnts = allen_compare(pth_update(kwargs['AtlasFile']), svlc_brain, impth, outline = True)
                cntlsts.append((nm, cnts))
                atlaspth = pth_update(kwargs['AtlasFile'])[kwargs['AtlasFile'].rfind('/'):] #use to compare with others
        
        if injsitethreshold:        
            #set path for post registered inj site
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            try:
                resultfld = [xx for xx in listdirfull(pth_update(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel))) if 'result' in xx and '.tif' in xx]; resultfld.sort()        
            except OSError:
                raise ValueError('Missing Elastix folder for {},\n   -Need to rerun elastix(step3)\n   -If used tools.util.update_inj_sites then set injsitethreshold=False and try again!'.format(injobj.brainname))
            
            #determine injection site via medianblurring and thresholding using postregistered brain
            if len(resultfld)>1:
                resultfl = resultfld[-2]
            else:
                resultfl = resultfld[0]
            
            sys.stdout.write('\n     loading injection site....'); sys.stdout.flush()
            if crop != False: 
                injstack = eval("tifffile.imread(resultfl).astype('uint16'){}".format(crop))
            else: 
                injstack = tifffile.imread(resultfl).astype('uint16') 
            for i in range(len(injstack)):
                injstack[i]=cv2.medianBlur(injstack[i].astype(np.uint16), 5)
            
            mxx = injstack.max()

            injstack[injstack<injsitethreshold] = 0; injstack=injstack*mxx

            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)

        
        elif not injsitethreshold:
            #load detected injsite (from tools.utils.update_inj_sites)
            sys.stdout.write('\n     generating injection site using transformed points...'); sys.stdout.flush()
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            ##load atlas to get correct shape and generate 'empty' stack of appropriate shape
            injstack = np.zeros(tifffile.imread(pth_update(kwargs['AtlasFile'])).shape).astype('uint8')
            #find points post transformed
            try:
                pntfld = [xx for xx in listdirfull(pth_update(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform'))) if injobj.brainname in xx and 'inj' in xx][0]
            except OSError:
                raise ValueError('Missing injection site post-transformed file in {},\n   -*usually* generated from tools.utils.update_inj_sites\n   -Need to rerun elastix(step3)'.format(pth_update(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform'))))                        
            #read points:
            points_file = [xx+'/injch/outputpoints.txt' for xx in listdirfull(pth_update(pntfld)) if 'atlas2reg2sig' in xx][0]
            with open(points_file, "rb") as f:                
                lines=f.readlines()
                f.close()
            #####populate post-transformed array of contour centers
            sys.stdout.write('\n     {} points detected'.format(len(lines)))
            arr=np.empty((len(lines), 3))    
            point_or_index = 'OutputIndexFixed' #typically use 'OutputPoint' instead but since this is only for visualization the approximation is OK!
            for i in range(len(lines)):        
                arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            for i in arr:
                injstack[i[2],i[1],i[0]] = 255 #populate positive pixels going from xyz to zyx
            if crop: injstack = eval("injstack{}".format(crop))      
            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)
    
        #flatten:
        sys.stdout.write('\n     max projecting each axis...'); sys.stdout.flush()
        loc = os.path.join(svlc_brain, 'tmp'); makedir(loc); rmlst.append(loc)
        projections=[os.path.join(loc, 'proj0.png'), os.path.join(loc, 'proj1.png'), os.path.join(loc, 'proj2.png')]        

        grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
        nametosave=[os.path.join(svlc_brain, 'tmp', "proj0_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj1_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj2_overlay.png")]        
                   
        #account for possible orientation of atlas        
        if atlaspth == pth_update(kwargs['AtlasFile'])[kwargs['AtlasFile'].rfind('/'):]: [cv2.imwrite(projections[x], np.max(injstack, axis=x)) for x in range(3)]
        if not atlaspth == pth_update(kwargs['AtlasFile'])[kwargs['AtlasFile'].rfind('/'):]: [tifffile.imwrite(projections[x], np.max(np.fliplr(injstack, axis=x))) for x in range(3)]
        #testing:
        if testing: [cv2.imwrite(projections[x], np.max(im, axis=x)) for x in range(3)]


        #find contours
        cntlst = []        
        for xx in range(len(projections)):
            im_gray = cv2.imread(projections[xx], 0)                      
            disp, center, cnt = detect_inj_site(im_gray.astype('uint8'), kernelsize = 2, threshold = 0.1, minimumarea=1)
            if xx==1: cnt = swap_cols(cnt, 0, 1)            
            if xx==2: cnt = swap_cols(cnt, 0, 1)                  
            cntlst.append([xx, cnt])
        cntlst = [cntlst[2], cntlst[1], cntlst[0]]
        cntdct.update(dict([(name, [kwargs, cntlst])]))


    
    ####

    ##inputs for big image
    grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
    nametosave=[os.path.join(svlc, "proj0_overlay.png"), os.path.join(svlc, "proj1_overlay.png"), os.path.join(svlc, "proj2_overlay.png")]        
    nametosave11=[os.path.join(svlc, "proj0_overlay11.png"), os.path.join(svlc, "proj1_overlay11.png"), os.path.join(svlc, "proj2_overlay11.png")]                    
    
    #learn to make a better colorscale
    #http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    if not colorlst: 
        clnm = colorlst        
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in ['r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']]
    else:
        clnm = colorlst                
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in colorlst]
        bgr = [tuple([x[2], x[1], x[0]]) for x in colorlst]
        
        
    legend=[]
    for x in range(3):
        im = load_n_make_bg(grayscale[x])#.astype('uint16')
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            cv2.drawContours(im, [cntlsts[xx][1][x]], 0, bgr[xx], 3)
            if bilateraloutlines:            
                if x==0 or x==1:
                    im2 = np.fliplr(im); im2 = im2.copy()
                    im[:, :im.shape[1]/2,:] = im2[:, :im.shape[1]/2,:] #im[:, im.shape[1]/2:,:] = im2[:, im.shape[1]/2:,:]
            if xx ==0: print ('\n\nCnt => {}   {}'.format(nm, bgr[xx]))
            if x==0: legend.append((nm, bgr[xx], clnm[xx]))
        if atlaspth[-25:-4] == 'reflectedhorizontally' and x == 1: im = np.flipud(im)
        cv2.imwrite(nametosave[x], im)            
        cv2.imwrite(nametosave11[x], im)
        '''if you want alphas on outline        
        #done to account for alphas
        im = load_n_make_bg(grayscale[x])
        blnkim = np.zeros(im.shape)
        outim = np.zeros((im.shape[0], im.shape[1], im.shape[2]+1)) #add alpha
        outim[...,0:3]=im
        outim[...,-1]=1
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            colors = colorlst[xx]
            cv2.drawContours(blnkim, [cntlsts[xx][1][x]], 0, colors, 3)
            if x==0: legend.append((nm, colors))
        nblnkim = np.zeros((im.shape[0], im.shape[1], im.shape[2]+1)) #add alpha
        nblnkim[...,0:3]=blnkim
        nblnkim[...,-1] = 1
        im = overlay_new(outim, nblnkim, alpha=1)
        if atlaspth[-25:-4] == 'reflectedhorizontally' and x == 1: im = np.flipud(im)
        pl.imsave(nametosave[x], im)            
        pl.imsave(nametosave11[x], im)  '''
    
    for x in range(3):
        #bgrnd            
        #im = make_bg(grayscale[x])
        #pl.ion(); pl.figure(); pl.imshow(im)
        im = cv2.imread(nametosave[x]) #change to pl, if alpha coloring outlines
        if im.shape[-1]>3:
            blnkim = np.zeros((im.shape[0], im.shape[1], im.shape[2]-1)) 
        else:
            blnkim = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
        tick=0
        for k,v in cntdct.iteritems():
            #if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)                    
            colors = bgr[tick+len(bglst)]; nmmm = clnm[tick+len(bglst)]; tick+=1         
            if x==0: sys.stdout.write('\n{}  {}...'.format(k, colors))
            if not v[1][x] == None: 
                if annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: blnkim = np.flipud(blnkim); blnkim = blnkim.copy()
                if annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: blnkim = np.fliplr(blnkim); blnkim = blnkim.copy()
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: blnkim = np.fliplr(blnkim); blnkim = blnkim.copy()
                cv2.fillPoly(blnkim, [np.asarray(v[1][x][1])], colors)
                if x == 0: legend.append((k, colors, nmmm))
                if annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: blnkim = np.flipud(blnkim); blnkim = blnkim.copy()
                if annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: blnkim = np.fliplr(blnkim); blnkim = blnkim.copy()
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: blnkim = np.fliplr(blnkim); blnkim = blnkim.copy()
            else:
                if x==0: sys.stdout.write('\n\n**************{} contour not found*****************\n\n'.format(k))

        blnkim = cv2.cvtColor(blnkim.astype('uint8'), cv2.COLOR_BGR2RGB)
        nblnkim = np.zeros(im.shape) #adding alpha
        nblnkim[...,0:3]=blnkim
        nblnkim[...,-1] = alpha
        nblnkim = (nblnkim - nblnkim.min()) / (nblnkim.max() - nblnkim.min())
        nblnkim[nblnkim<0.1]=0
        im = im+nblnkim        
        #im = overlay_new(im, nblnkim, alpha=alpha)
        #pl.ion(); pl.figure(); pl.imshow(im)
        #if x == 0 or x == 1: im = np.fliplr(im); im = im.copy() #might need this to flip
        pl.imsave(nametosave[x], im)

    
        

    #make overlay and add legend
    depth.layout(nametosave[0], nametosave[1], nametosave[2], svlc) ###might need 1,0,2 for some files
    im = cv2.imread(os.path.join(svlc, 'summary.png'), 1); rmlst.append(os.path.join(svlc, 'summary.png'))
    for xx in range(len(legend)):        
        if xx < len(legend): nm = legend[xx][0]
        if len(nm) > 35: nm = nm[:35]
        #if xx < len(bglst): cv2.putText(im, nm,(55, 510+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2)
        if xx < len(bglst): cv2.putText(im, nm,(55+(xx*150), 510), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2) #TP think this is 'bottom' legend showing contours
        if xx >= len(bglst): cv2.putText(im, nm,(55, 200+(xx*28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2)
    #brainsvlc = os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(brainsvlc)
    cv2.imwrite(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname)), im)
    sys.stdout.write('\n\nLegend: {}\n\n'.format(legend))
    sys.stdout.write('\n   Finished Injection Site - saved in {}\n'.format(svlc)); sys.stdout.flush()
    
    #remove files:
    if cleanup: [removedir(xx) for xx in rmlst]
    
    return
#%%
def generate_inj_fig_multi(pths, svlc, bglst=False, crop=False, injsitethreshold = False, colorlst = False, cleanup=True, bilateraloutlines=True, testing = False, show=False):    
    '''Function to generate outlines of structres of interest WITHOUT ALPHA*****
    
    NOTE THIS REQUIRES THAT "STEP3" elastix registration to be run or tools.utils.update_inj_sites
    
    Inputs
    -------------
    pthlst = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = savelocation
    bglst = structures to outline in the following format: [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726]]    
    crop = 
          cerebellum: '[:,390:,:]'
          caudal midbrain: '[:,300:415,:]'
          midbrain: '[:,215:415,:]'
          thalamus: '[:,215:345,:]'
          anterior cortex: '[:,:250,:]'
    injsitethreshold = 
                int  --> use this function to detect injection site via median blurring and thresholding (value of this parameter)
                false --> use contours found by light sheet package (from tools.utils.update_inj_sites)
    colorlst = ['darksalmon', 'lightgreen', 'skyblue', 'b', 'r', 'g', 'm', 'y', 'c', 'darkturquoise', 'lime', 'firebrick', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']
    cleanup = True to delete intermediate files that are generated
    bilateraloutlines=True; if false will only display single sided outlines
    testing = False used to make sure orientation is correct - THIS WILL STOP THE DISPLAY OF ACTUAL INJECTION SITES, literally just for testing
    show = if True use sitk to show disp
    
    '''
    makedir(svlc)
    rmlst = []
    cntdct = {}
    
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct
    
    #load kwargs
    for name, pth in pths.iteritems():
        ##inputs:    
        kwargs={}  
        try:        
            with open(os.path.join(pth, 'param_dict.p' ), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
        except IOError:
            with open(os.path.join(pth, 'param_dict.p'), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
                
        if directorydeterminer() == '/home/wanglab/': kwargs.update(kwargs)
        
        svlc_brain=os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(svlc_brain); rmlst.append(svlc_brain)
        
        #set up allen/ann stuff
        sys.stdout.write('\n\nStarting {}...'.format(name)); sys.stdout.flush()
        annfl = kwargs['annotationfile']
        sys.stdout.write('\n     Annotation File: {}'.format(annfl[annfl.rfind('/')+1:])); sys.stdout.flush()        
        #ann = sitk.ReadImage(pth_update(kwargs['annotationfile']))

        #bglst        
        if not bglst: bglst = [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726], ['CrI_II', 1017, 1056, 10677,10676,10675, 1064,10680,10679,10678]]
                    
        #Generate contours if first brain to process: 
        if pth == pths.values()[0]:        
            #make outline for each structure in bglst
            sys.stdout.write('\n     generating outlines for bglst structures (time consuming, only done once)...'); sys.stdout.flush()            
            cntlsts = []
            for lst in bglst:
                nm = lst[0]        
                if crop != False: 
                    im = eval("isolate_structures(kwargs['annotationfile'], *lst[1:]){}".format(crop))
                    #ann = eval("sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))".format(crop))
                else:
                    im = isolate_structures(kwargs['annotationfile'], *lst[1:])
                    #ann = sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))
                    
        
                
                svlc2=os.path.join(svlc_brain, nm)
                makedir(svlc2); rmlst.append(svlc2)
                impth=os.path.join(svlc2, str(kwargs['volumes'][0].brainname)+'.tif')
                tifffile.imsave(impth, im.astype('uint8'))
                
                allen = os.path.join(svlc_brain, 'allenatlas'); makedir(allen); rmlst.append(allen)
                if crop != False: 
                    croppedAtlasFile = os.path.join(allen, 'cropped_atlas.tif')   
                    eval("tifffile.imsave(croppedAtlasFile, tifffile.imread(kwargs['AtlasFile']){})".format(crop))
                    cnts = allen_compare(croppedAtlasFile, svlc_brain, impth, outline = True)
                else:
                    cnts = allen_compare(kwargs['AtlasFile'], svlc_brain, impth, outline = True)
                cntlsts.append((nm, cnts))
                atlaspth = kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):] #use to compare with others
        
        ###Injection Site    
        sys.stdout.write('\n     processing injection site...'); sys.stdout.flush()            

        if injsitethreshold:        
            #set path for post registered inj site
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            try:
                resultfld = [xx for xx in listdirfull(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel)) if 'result' in xx and '.tif' in xx]; resultfld.sort()        
            except OSError:
                raise ValueError('Missing Elastix folder for {},\n   -Need to rerun elastix(step3)\n   -If used tools.util.update_inj_sites then set injsitethreshold=False and try again!'.format(injobj.brainname))
            
            #determine injection site via medianblurring and thresholding using postregistered brain
            if len(resultfld)>1:
                resultfl = resultfld[-2]
            else:
                resultfl = resultfld[0]
            
            sys.stdout.write('\n     loading injection site....'); sys.stdout.flush()
            if crop != False: 
                injstack = eval("tifffile.imread(resultfl).astype('uint16'){}".format(crop))
            else: 
                injstack = tifffile.imread(resultfl).astype('uint16') 
            for i in range(len(injstack)):
                injstack[i]=cv2.medianBlur(injstack[i].astype(np.uint16), 5)
            
            mxx = injstack.max()

            injstack[injstack<injsitethreshold] = 0; injstack=injstack*mxx

            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)

        
        elif not injsitethreshold:
            #load detected injsite (from tools.utils.update_inj_sites)
            sys.stdout.write('\n     generating injection site using transformed points...'); sys.stdout.flush()
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            ##load atlas to get correct shape and generate 'empty' stack of appropriate shape
            injstack = np.zeros(tifffile.imread(kwargs['AtlasFile']).shape).astype('uint8')
            #find points post transformed
            try:
                pntfld = [xx for xx in listdirfull(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform')) if injobj.brainname in xx and 'inj' in xx][0]
            except OSError:
                raise ValueError('Missing injection site post-transformed file in {},\n   -*usually* generated from tools.utils.update_inj_sites\n   -Need to rerun elastix(step3)'.format(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform')))
            #read points:
            points_file = [xx+'/injch/outputpoints.txt' for xx in listdirfull(pntfld) if 'atlas2reg2sig' in xx][0]
            with open(points_file, "rb") as f:                
                lines=f.readlines()
                f.close()
            #####populate post-transformed array of contour centers
            sys.stdout.write('\n     {} points detected'.format(len(lines)))
            arr=np.empty((len(lines), 3))    
            point_or_index = 'OutputIndexFixed' #typically use 'OutputPoint' instead but since this is only for visualization the approximation is OK!
            for i in range(len(lines)):        
                arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            arr = filter_overinterpolated_pixels(arr, **kwargs)            
            for i in arr:
                injstack[i[2],i[1],i[0]] = 255 #populate positive pixels going from xyz to zyx
            if crop: injstack = eval("injstack{}".format(crop))      
            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)
    
        #flatten:
        sys.stdout.write('\n     max projecting each axis...'); sys.stdout.flush()
        loc = os.path.join(svlc_brain, 'tmp'); makedir(loc); rmlst.append(loc)
        projections=[os.path.join(loc, 'proj0.png'), os.path.join(loc, 'proj1.png'), os.path.join(loc, 'proj2.png')]        

        grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
        nametosave=[os.path.join(svlc_brain, 'tmp', "proj0_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj1_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj2_overlay.png")]        
                   
        #account for possible orientation of atlas        
        if atlaspth == kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):]: [cv2.imwrite(projections[x], np.max(injstack, axis=x)) for x in range(3)]
        if not atlaspth == kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):]: [cv2.imwrite(projections[x], np.max(np.fliplr(injstack), axis=x)) for x in range(3)]
        #testing:
        if testing: [cv2.imwrite(projections[x], np.max(im, axis=x)) for x in range(3)]


        #find contours
        cntlst = []        
        for xx in range(len(projections)):
            im_gray = cv2.imread(projections[xx], 0)
            disp, center, cnt = detect_inj_site(im_gray.astype('uint8'), kernelsize = 2, threshold = 0.1, minimumarea=1)
            if show: sitk.Show(sitk.GetImageFromArray(disp))
            if xx==1: cnt = swap_cols(cnt, 0, 1)            
            if xx==2: cnt = swap_cols(cnt, 0, 1)                  
            cntlst.append([xx, cnt])
        cntlst = [cntlst[2], cntlst[1], cntlst[0]]
        #cntlst = [cntlst[0], cntlst[1], cntlst[2]]
        cntdct.update(dict([(name, [kwargs, cntlst])]))
    
    ####

    ##inputs for big image
    grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
    nametosave=[os.path.join(svlc, "proj0_overlay.png"), os.path.join(svlc, "proj1_overlay.png"), os.path.join(svlc, "proj2_overlay.png")]        
    nametosave11=[os.path.join(svlc, "proj0_overlay11.png"), os.path.join(svlc, "proj1_overlay11.png"), os.path.join(svlc, "proj2_overlay11.png")]                    
    
    #learn to make a better colorscale
    #http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    if not colorlst: 
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in ['r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b', 'r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'r', 'c', 'b']]
        clnm = colorlst   
    else:
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in colorlst]
        clnm = colorlst                
    bgr = [tuple([x[2], x[1], x[0]]) for x in colorlst]

    #add outlines first
    legend=[]
    for x in range(3):
        im = load_n_make_bg(grayscale[x])#.astype('uint16')
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            cv2.drawContours(im, [cntlsts[xx][1][x]], 0, bgr[xx], 3)
            if bilateraloutlines:            
                if x==0 or x==1:
                    im2 = np.fliplr(im); im2 = im2.copy()
                    im[:, :im.shape[1]/2,:] = im2[:, :im.shape[1]/2,:]
            if xx ==0: print ('\n\nCnt => {}   {}'.format(nm, bgr[xx]))
            if x==0: legend.append((nm, bgr[xx], clnm[xx]))
        if atlaspth[-25:-4] == 'reflectedhorizontally' and x == 1: im = np.flipud(im)
        cv2.imwrite(nametosave[x], im)            
        cv2.imwrite(nametosave11[x], im)
    
    for x in range(3):
        #bgrnd            
        #im = make_bg(grayscale[x])
        #pl.ion(); pl.figure(); pl.imshow(im)
        im = cv2.imread(nametosave[x])
        tick=0
        for k,v in cntdct.iteritems():
            #if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)                    
            colors = bgr[tick+len(bglst)]; nmmm = clnm[tick+len(bglst)]; tick+=1
            if x==0: sys.stdout.write('\n{}  {}...'.format(k, colors))
            if not v[1][x] == None: 
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: im = np.flipud(im); im = im.copy()
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: im = np.fliplr(im); im = im.copy()
                if not v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()

                cv2.fillPoly(im, [np.asarray(v[1][x][1])], colors)
                if not x == 0: legend.append((k, colors, nmmm))
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: im = np.flipud(im); im = im.copy()
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: im = np.fliplr(im); im = im.copy()
                if not v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()
            else:
                if x==0: sys.stdout.write('\n\n**************{} contour not found*****************\n\n'.format(k))
 
        #im = overlay_new(im, nblnkim, alpha=alpha)
        #pl.ion(); pl.figure(); pl.imshow(im)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if x == 0 or x == 1: im = np.fliplr(im); im = im.copy()
        cv2.imwrite(nametosave[x], im)
        

    
        

    #make overlay and add legend
    depth.layout(nametosave[0], nametosave[1], nametosave[2], svlc) ###might need 1,0,2 for some files
    im = cv2.imread(os.path.join(svlc, 'summary.png'), 1); rmlst.append(os.path.join(svlc, 'summary.png'))
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for xx in range(len(legend)):        
        if xx < len(legend): nm = legend[xx][0]
        if len(nm) > 35: nm = nm[:35]
        #if xx < len(bglst): cv2.putText(im, nm,(55, 510+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2) #TP think this is 'bottom' legend showing contours
        if xx < len(bglst): cv2.putText(im, nm,(55+(xx*150), 510), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2) #TP think this is 'bottom' legend showing contours
        if xx < len(bglst): print ('\n\ncv2.putText=> where nm={} legend[xx]={}'.format(nm, legend[xx][1]))
        if xx >= len(bglst): cv2.putText(im, nm,(55, 200+(xx*28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2)
    #brainsvlc = os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(brainsvlc)
    ###NOTE MIGHT NEED TO FIND A WAY TO INCREASE DPI HERE
    cv2.imwrite(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname)), im)
    pl.imsave(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname+'450dpi')), im, dpi=450)
    sys.stdout.write('\n\nLegend: {}\n\n'.format(legend))
    sys.stdout.write('\n   Finished Injection Site - saved in {}\n'.format(svlc)); sys.stdout.flush()
    
    #remove files:
    if cleanup: [removedir(xx) for xx in rmlst]
    
    return


#%%
def generate_inj_fig_multi_old(pths, svlc, bglst=False, crop=False, injsitethreshold = False, colorlst = False, cleanup=True, bilateraloutlines=True, testing = False, show=False):    
    '''Function to generate outlines of structres of interest WITHOUT ALPHA*****
    
    NOTE THIS REQUIRES THAT "STEP3" elastix registration to be run or tools.utils.update_inj_sites
    
    Inputs
    -------------
    pthlst = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = savelocation
    bglst = structures to outline in the following format: [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726]]    
    crop = 
          cerebellum: '[:,390:,:]'
          caudal midbrain: '[:,300:415,:]'
          midbrain: '[:,215:415,:]'
          thalamus: '[:,215:345,:]'
          anterior cortex: '[:,:250,:]'
    injsitethreshold = 
                int  --> use this function to detect injection site via median blurring and thresholding (value of this parameter)
                false --> use contours found by light sheet package (from tools.utils.update_inj_sites)
    colorlst = ['darksalmon', 'lightgreen', 'skyblue', 'b', 'r', 'g', 'm', 'y', 'c', 'darkturquoise', 'lime', 'firebrick', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']
    cleanup = True to delete intermediate files that are generated
    bilateraloutlines=True; if false will only display single sided outlines
    testing = False used to make sure orientation is correct - THIS WILL STOP THE DISPLAY OF ACTUAL INJECTION SITES, literally just for testing
    show = if True use sitk to show disp
    
    '''
    makedir(svlc)
    rmlst = []
    cntdct = {}
    
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct
    
    #load kwargs
    for name, pth in pths.iteritems():
        ##inputs:    
        kwargs={}  
        try:        
            with open(os.path.join(pth, 'param_dict.p' ), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
        except IOError:
            with open(os.path.join(pth, 'param_dict.p'), 'rb') as pckl:
                kwargs.update(pickle.load(pckl))
                pckl.close()
                
        if directorydeterminer() == '/home/wanglab/': kwargs.update(kwargs)
        
        svlc_brain=os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(svlc_brain); rmlst.append(svlc_brain)
        
        #set up allen/ann stuff
        sys.stdout.write('\n\nStarting {}...'.format(name)); sys.stdout.flush()
        annfl = kwargs['annotationfile']
        sys.stdout.write('\n     Annotation File: {}'.format(annfl[annfl.rfind('/')+1:])); sys.stdout.flush()        
        #ann = sitk.ReadImage(pth_update(kwargs['annotationfile']))

        #bglst        
        if not bglst: bglst = [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726], ['CrI_II', 1017, 1056, 10677,10676,10675, 1064,10680,10679,10678]]
                    
        #Generate contours if first brain to process: 
        if pth == pths.values()[0]:        
            #make outline for each structure in bglst
            sys.stdout.write('\n     generating outlines for bglst structures (time consuming, only done once)...'); sys.stdout.flush()            
            cntlsts = []
            for lst in bglst:
                nm = lst[0]        
                if crop != False: 
                    im = eval("isolate_structures(kwargs['annotationfile'], *lst[1:]){}".format(crop))
                    #ann = eval("sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))".format(crop))
                else:
                    im = isolate_structures(kwargs['annotationfile'], *lst[1:])
                    #ann = sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))
                    
        
                
                svlc2=os.path.join(svlc_brain, nm)
                makedir(svlc2); rmlst.append(svlc2)
                impth=os.path.join(svlc2, str(kwargs['volumes'][0].brainname)+'.tif')
                tifffile.imsave(impth, im.astype('uint8'))
                
                allen = os.path.join(svlc_brain, 'allenatlas'); makedir(allen); rmlst.append(allen)
                if crop != False: 
                    croppedAtlasFile = os.path.join(allen, 'cropped_atlas.tif')   
                    eval("tifffile.imsave(croppedAtlasFile, tifffile.imread(kwargs['AtlasFile']){})".format(crop))
                    cnts = allen_compare(croppedAtlasFile, svlc_brain, impth, outline = True)
                else:
                    cnts = allen_compare(kwargs['AtlasFile'], svlc_brain, impth, outline = True)
                cntlsts.append((nm, cnts))
                atlaspth = kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):] #use to compare with others
        
        ###Injection Site    
        sys.stdout.write('\n     processing injection site...'); sys.stdout.flush()            

        if injsitethreshold:        
            #set path for post registered inj site
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            try:
                resultfld = [xx for xx in listdirfull(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel)) if 'result' in xx and '.tif' in xx]; resultfld.sort()        
            except OSError:
                raise ValueError('Missing Elastix folder for {},\n   -Need to rerun elastix(step3)\n   -If used tools.util.update_inj_sites then set injsitethreshold=False and try again!'.format(injobj.brainname))
            
            #determine injection site via medianblurring and thresholding using postregistered brain
            if len(resultfld)>1:
                resultfl = resultfld[-2]
            else:
                resultfl = resultfld[0]
            
            sys.stdout.write('\n     loading injection site....'); sys.stdout.flush()
            if crop != False: 
                injstack = eval("tifffile.imread(resultfl).astype('uint16'){}".format(crop))
            else: 
                injstack = tifffile.imread(resultfl).astype('uint16') 
            for i in range(len(injstack)):
                injstack[i]=cv2.medianBlur(injstack[i].astype(np.uint16), 5)
            
            mxx = injstack.max()

            injstack[injstack<injsitethreshold] = 0; injstack=injstack*mxx

            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)

        
        elif not injsitethreshold:
            #load detected injsite (from tools.utils.update_inj_sites)
            sys.stdout.write('\n     generating injection site using transformed points...'); sys.stdout.flush()
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            ##load atlas to get correct shape and generate 'empty' stack of appropriate shape
            injstack = np.zeros(tifffile.imread(kwargs['AtlasFile']).shape).astype('uint8')
            #find points post transformed
            try:
                pntfld = [xx for xx in listdirfull(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform')) if injobj.brainname in xx and 'inj' in xx][0]
            except OSError:
                raise ValueError('Missing injection site post-transformed file in {},\n   -*usually* generated from tools.utils.update_inj_sites\n   -Need to rerun elastix(step3)'.format(os.path.join(kwargs['outputdirectory'], 'elastix_inverse_transform')))                        
            #read points:
            points_file = [xx+'/injch/outputpoints.txt' for xx in listdirfull(pntfld) if 'atlas2reg2sig' in xx][0]
            with open(points_file, "rb") as f:                
                lines=f.readlines()
                f.close()
            #####populate post-transformed array of contour centers
            sys.stdout.write('\n     {} points detected'.format(len(lines)))
            arr=np.empty((len(lines), 3))    
            point_or_index = 'OutputIndexFixed' #typically use 'OutputPoint' instead but since this is only for visualization the approximation is OK!
            for i in range(len(lines)):        
                arr[i,...]=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            arr = filter_overinterpolated_pixels(arr, **kwargs)            
            for i in arr:
                injstack[i[2],i[1],i[0]] = 255 #populate positive pixels going from xyz to zyx
            if crop: injstack = eval("injstack{}".format(crop))      
            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)
    
        #flatten:
        sys.stdout.write('\n     max projecting each axis...'); sys.stdout.flush()
        loc = os.path.join(svlc_brain, 'tmp'); makedir(loc); rmlst.append(loc)
        projections=[os.path.join(loc, 'proj0.png'), os.path.join(loc, 'proj1.png'), os.path.join(loc, 'proj2.png')]        

        grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
        nametosave=[os.path.join(svlc_brain, 'tmp', "proj0_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj1_overlay.png"), os.path.join(svlc_brain, 'tmp', "proj2_overlay.png")]        
                   
        #account for possible orientation of atlas        
        if atlaspth == kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):]: [cv2.imwrite(projections[x], np.max(injstack, axis=x)) for x in range(3)]
        if not atlaspth == kwargs['AtlasFile'][kwargs['AtlasFile'].rfind('/'):]: [cv2.imwrite(projections[x], np.max(np.fliplr(injstack), axis=x)) for x in range(3)]
        #testing:
        if testing: [cv2.imwrite(projections[x], np.max(im, axis=x)) for x in range(3)]


        #find contours
        cntlst = []        
        for xx in range(len(projections)):
            im_gray = cv2.imread(projections[xx], 0)
            disp, center, cnt = detect_inj_site(im_gray.astype('uint8'), kernelsize = 2, threshold = 0.1, minimumarea=1)
            if show: sitk.Show(sitk.GetImageFromArray(disp))
            if xx==1: cnt = swap_cols(cnt, 0, 1)            
            if xx==2: cnt = swap_cols(cnt, 0, 1)                  
            cntlst.append([xx, cnt])
        cntlst = [cntlst[2], cntlst[1], cntlst[0]]
        #cntlst = [cntlst[0], cntlst[1], cntlst[2]]
        cntdct.update(dict([(name, [kwargs, cntlst])]))
    
    ####

    ##inputs for big image
    grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
    nametosave=[os.path.join(svlc, "proj0_overlay.png"), os.path.join(svlc, "proj1_overlay.png"), os.path.join(svlc, "proj2_overlay.png")]        
    nametosave11=[os.path.join(svlc, "proj0_overlay11.png"), os.path.join(svlc, "proj1_overlay11.png"), os.path.join(svlc, "proj2_overlay11.png")]                    
    
    #learn to make a better colorscale
    #http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    if not colorlst: 
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in ['r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']]
        clnm = colorlst   
    else:
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in colorlst]
        clnm = colorlst                
    bgr = [tuple([x[2], x[1], x[0]]) for x in colorlst]

    #add outlines first
    legend=[]
    for x in range(3):
        im = load_n_make_bg(grayscale[x])#.astype('uint16')
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            cv2.drawContours(im, [cntlsts[xx][1][x]], 0, bgr[xx], 3)
            if bilateraloutlines:            
                if x==0 or x==1:
                    im2 = np.fliplr(im); im2 = im2.copy()
                    im[:, :im.shape[1]/2,:] = im2[:, :im.shape[1]/2,:]
            if xx ==0: print ('\n\nCnt => {}   {}'.format(nm, bgr[xx]))
            if x==0: legend.append((nm, bgr[xx], clnm[xx]))
        if atlaspth[-25:-4] == 'reflectedhorizontally' and x == 1: im = np.flipud(im)
        cv2.imwrite(nametosave[x], im)            
        cv2.imwrite(nametosave11[x], im)
    
    for x in range(3):
        #bgrnd            
        #im = make_bg(grayscale[x])
        #pl.ion(); pl.figure(); pl.imshow(im)
        im = cv2.imread(nametosave[x])
        tick=0
        for k,v in cntdct.iteritems():
            #if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)                    
            colors = bgr[tick+len(bglst)]; nmmm = clnm[tick+len(bglst)]; tick+=1
            if x==0: sys.stdout.write('\n{}  {}...'.format(k, colors))
            if not v[1][x] == None: 
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: im = np.flipud(im); im = im.copy()
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: im = np.fliplr(im); im = im.copy()
                if not v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()

                cv2.fillPoly(im, [np.asarray(v[1][x][1])], colors)
                if not x == 0: legend.append((k, colors, nmmm))
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 1: im = np.flipud(im); im = im.copy()
                if not annfl[annfl.rfind('/')+1:] == 'annotation_25_ccf2015.nrrd' and x == 2: im = np.fliplr(im); im = im.copy()
                if not v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()
            else:
                if x==0: sys.stdout.write('\n\n**************{} contour not found*****************\n\n'.format(k))
 
        #im = overlay_new(im, nblnkim, alpha=alpha)
        #pl.ion(); pl.figure(); pl.imshow(im)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if x == 0 or x == 1: im = np.fliplr(im); im = im.copy()
        cv2.imwrite(nametosave[x], im)
        

    
        

    #make overlay and add legend
    depth.layout(nametosave[0], nametosave[1], nametosave[2], svlc) ###might need 1,0,2 for some files
    im = cv2.imread(os.path.join(svlc, 'summary.png'), 1); rmlst.append(os.path.join(svlc, 'summary.png'))
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for xx in range(len(legend)):        
        if xx < len(legend): nm = legend[xx][0]
        if len(nm) > 35: nm = nm[:35]
        #if xx < len(bglst): cv2.putText(im, nm,(55, 510+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2) #TP think this is 'bottom' legend showing contours
        if xx < len(bglst): cv2.putText(im, nm,(55+(xx*150), 510), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2) #TP think this is 'bottom' legend showing contours
        if xx < len(bglst): print ('\n\ncv2.putText=> where nm={} legend[xx]={}'.format(nm, legend[xx][1]))
        if xx >= len(bglst): cv2.putText(im, nm,(55, 200+(xx*28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, legend[xx][1] , 2)
    #brainsvlc = os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(brainsvlc)
    ###NOTE MIGHT NEED TO FIND A WAY TO INCREASE DPI HERE
    cv2.imwrite(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname)), im)
    pl.imsave(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname+'450dpi')), im, dpi=450)
    sys.stdout.write('\n\nLegend: {}\n\n'.format(legend))
    sys.stdout.write('\n   Finished Injection Site - saved in {}\n'.format(svlc)); sys.stdout.flush()
    
    #remove files:
    if cleanup: [removedir(xx) for xx in rmlst]
    
    return
   
#%%
   
   
   #OLD
#%%
def generate_inj_fig_multi_old_old(pths, svlc, bglst=False, cerebellum=True, midbrain=False, threshold = 10000, detectinjsite = True, colorlst = False, cleanup=True):    
    '''Function to generate outlines of structres of interest
    
    Inputs
    -------------
    pthlst = list of paths or dictionary to output of lightsheet package; if dictionary: [name:path]
    svlc = savelocation
    '''
    makedir(svlc)
    rmlst = []
    cntdct = {}
    
    #determine if input was a list or dictionary
    if type(pths) == list:
        dct={}
        for pth in pths:
            dct.update(dict([(pth[pth.rfind('/')+1:], pth)]))
        pths = dct
    
    #run
    for name, pth in pths.iteritems():
        ##inputs:    
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
        
        svlc_brain=os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(svlc_brain); rmlst.append(svlc_brain)
        #load
        sys.stdout.write('\n\nStarting {}...'.format(name)); sys.stdout.flush()
        #allen_id_table=pd.read_excel(pth_update(os.path.join(kwargs['volumes'][0].packagedirectory, 'supp_files/allen_id_table.xlsx'))) ##use for determining neuroanatomical locations according to allen
        ann = sitk.ReadImage(pth_update(kwargs['annotationfile']))
        if not bglst: bglst = [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726], ['CrI_II', 1017, 1056, 10677,10676,10675, 1064,10680,10679,10678]]
                    
        if pth == pths.values()[0]:        
            #make outline for each structure in bglst
            sys.stdout.write('\n     generating outlines for bglst structures (time consuming, only done once)...'); sys.stdout.flush()            
            cntlsts = []
            for lst in bglst:
                nm = lst[0]        
                if cerebellum: 
                    im = isolate_structures(pth_update(kwargs['annotationfile']), *lst[1:])[:,390:,:]
                    ann = sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))[:,390:,:]
                else:
                    im = isolate_structures(pth_update(kwargs['annotationfile']), *lst[1:])
                    ann = sitk.GetArrayFromImage(sitk.ReadImage(pth_update(kwargs['annotationfile'])))
        
                
                svlc2=os.path.join(svlc_brain, nm)
                makedir(svlc2); rmlst.append(svlc2)
                impth=os.path.join(svlc2, str(pth_update(kwargs['volumes'][0].brainname)+'.tif'))    
                tifffile.imsave(impth, im.astype('uint8'))
                
                allen = os.path.join(svlc_brain, 'allenatlas'); makedir(allen); rmlst.append(allen)
                if cerebellum: 
                    CbAtlasFile = os.path.join(allen, 'cb_atlas.tif')   
                    tifffile.imsave(CbAtlasFile, tifffile.imread(pth_update(kwargs['AtlasFile']))[:,390:,:])
                    cnts = allen_compare(CbAtlasFile, svlc_brain, impth, outline = True)
                else:
                    cnts = allen_compare(pth_update(kwargs['AtlasFile']), svlc_brain, impth, outline = True)
                cntlsts.append((nm, cnts))
        
        ###Injection Site    
        sys.stdout.write('\n     processing injection site...'); sys.stdout.flush()            
        
        if not detectinjsite:        
            #load and clean up injection site
            sys.stdout.write('\n     loading injection site...'); sys.stdout.flush()
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            if cerebellum: injstack = tifffile.imread(pth_update(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel, 'result.tif'))).astype('uint16')[:,390:,:]
            if not cerebellum: injstack = tifffile.imread(pth_update(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel, 'result.tif'))).astype('uint16') 
            for i in range(len(injstack)):
                injstack[i]=cv2.medianBlur(injstack[i].astype(np.uint16), 5)
            mxx = injstack.max()
            #injstack = (injstack - injstack.min()) / (injstack.max() - injstack.min())    
            injstack[injstack<threshold] = 0; injstack=injstack*mxx
            #try: 
            #    injim = np.swapaxes(injim, *kwargs['swapaxes'])
            #except:
            #    pass
            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); del injstack; rmlst.append(injtif)
        
        elif detectinjsite: #hopefully functional now
            #load detected injsite
            sys.stdout.write('\n     loading injection site...'); sys.stdout.flush()
            injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
            impth = [xx for xx in listdirfull(pth_update(os.path.join(kwargs['outputdirectory'], 'injection'))) if '.tif' in xx or '.npy' in xx][0]


            #load and get injection site to correct shape
            sys.stdout.write('\n   Loading injection site....'); sys.stdout.flush()
            if not 'resampledforelastix' in impth and '.tif' in impth: injstack = resample_par(6, impth, kwargs['AtlasFile'], svlocname=None, singletifffile=True, resamplefactor=1.3)
            elif not 'resampledforelastix' in impth and '.npy' in impth:
                injstack = np.load(impth); atlshp = tuple(tifffile.imread(pth_update(kwargs['AtlasFile'])).shape)
                rescale = tuple([float(atlshp[xx]/injstack.shape[xx]) for xx in range(3)]) #zyx of horizontal
                injstack = zoom(injstack , rescale)
            elif impth[-4:] == '.npy': injstack = np.load(impth)
            elif impth[-4:] == '.tif': injstack = tifffile.imread(impth)

            injtif = os.path.join(svlc_brain, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); rmlst.append(injtif)
    
        #colorcode:
        sys.stdout.write('\n     colorcoding...'); sys.stdout.flush()
        loc = os.path.join(svlc_brain, 'tmp'); makedir(loc); rmlst.append(loc)
        depth.colorcode(injtif, loc)
        color=[os.path.join(loc, 'proj0.png'), os.path.join(loc, 'proj1.png'), os.path.join(loc, 'proj2.png')]
        #find contours
        cntlst = []        
        for xx in color:
            im_gray = cv2.imread(xx, 0)                      
            disp, center, cnt = detect_inj_site(im_gray.astype('uint8'), kernelsize = 2, threshold = 0.1, minimumarea=1); cntlst.append(cnt)
        cntdct.update(dict([(name, [kwargs, cntlst])]))


    
    ####

    ##inputs for big image
    grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
    nametosave=[os.path.join(svlc, "proj0_overlay.png"), os.path.join(svlc, "proj1_overlay.png"), os.path.join(svlc, "proj2_overlay.png")]        
    nametosave11=[os.path.join(svlc, "proj0_overlay11.png"), os.path.join(svlc, "proj1_overlay11.png"), os.path.join(svlc, "proj2_overlay11.png")]                    
    
    #learn to make a better colorscale
    #http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    if not colorlst: 
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in ['r', 'b', 'g', 'darkturquoise', 'lime', 'firebrick', 'midnightblue', 'cyan', 'violet', 'darkgreen', 'g', 'r', 'b']]
    else:
        colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in colorlst]
            
    legend=[]; missing=[]
    for x in range(3):
        depth.overlay(grayscale[x], color[x], nametosave[x], alpha=0.95)
        im = cv2.imread(nametosave[x], 1)
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            colors = colorlst[xx]
            cv2.drawContours(im, [cntlsts[xx][1][x]], 0, colors, 3)
            if x==0: legend.append((nm, colors))
        if kwargs['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)
        cv2.imwrite(nametosave[x], im)            
        cv2.imwrite(nametosave11[x], im)  
    
    for x in range(3):
        #bgrnd            
        #im = make_bg(grayscale[x])
        #pl.ion(); pl.figure(); pl.imshow(im)
        im = cv2.imread(nametosave[x])
        tick=0
        for k,v in cntdct.iteritems():
            #if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)                    
            colors = colorlst[tick+3]; tick+=1
            if x==0: sys.stdout.write('\n{}  {}...'.format(k, colors))
            if not v[1][x] == None: 
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 1: im = np.flipud(im); im = im.copy()
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()
                cv2.fillPoly(im, [np.asarray(v[1][x])], colors)
                if x == 0: legend.append((k, colors))
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 1: im = np.flipud(im); im = im.copy()
                if v[0]['annotationfile'][-26:-5] == 'reflectedhorizontally' and x == 2: im = np.fliplr(im); im = im.copy()
            else:
                if x==0: sys.stdout.write('\n\n**************{} contour not found*****************\n\n'.format(k))
                if x==0: missing.append(k)
            
        #pl.ion(); pl.figure(); pl.imshow(im)
        cv2.imwrite(nametosave[x], im)
    

    #make overlay and add legend
    depth.layout(nametosave[0], nametosave[1], nametosave[2], svlc) ###might need 1,0,2 for some files
    im = cv2.imread(os.path.join(svlc, 'summary.png'), 1); rmlst.append(os.path.join(svlc, 'summary.png'))
    for xx in range(len(legend)+len(missing)):        
        if xx < len(legend): nm = legend[xx][0]
        if len(nm) > 35: nm = nm[:35]
        if xx < len(legend): clr = legend[xx][1]
        if xx < 3: cv2.putText(im, nm,(55, 510+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr , 2)
        if xx >= 3: cv2.putText(im, nm,(55, 180+(xx*28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr , 2)
        if xx >= len(legend): cv2.putText(im, 'MISSING: {}'.format(missing[xx + 1 - len(legend)]) , (15, 130+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr , 2)
    #brainsvlc = os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(brainsvlc)
    cv2.imwrite(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname)), im)
    sys.stdout.write('\n\nLegend: {}\n\n'.format(legend))
    sys.stdout.write('\n   Finished Injection Site - saved as {}\n'.format(os.path.join(svlc, 'summary.png'))); sys.stdout.flush()
    
    #remove files:
    if cleanup: [removedir(xx) for xx in rmlst]
    
    return

#%%
    
def generate_inj_fig(pth, svlc, bglst=False, cerebellum=True, threshold = 10000, detectinjsite=True, cleanup=True):    
    '''Function to generate outlines of structres of interest
    
    Inputs
    -------------
    pth = path to output of lightsheet package 
    svlc = savelocation
    detectedinjsite = if true use lightsheet packages detection, if false threshold inj site    
    threshold = see above (uint16)
    '''
    
    ##inputs:    
    if directorydeterminer() == '/home/wanglab/': param_dict = 'param_dict_local.p'
    if not directorydeterminer() == '/home/wanglab/': param_dict = 'param_dict.p'    
    kwargs={}  
    with open(os.path.join(pth, param_dict), 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()
    if directorydeterminer() == '/home/wanglab/': kwargs.update(pth_update(kwargs))
    
    #load
    sys.stdout.write('Starting {}...'.format(kwargs['volumes'][0].brainname)); sys.stdout.flush()
    allen_id_table=pd.read_excel(os.path.join(kwargs['volumes'][0].packagedirectory, 'supp_files/allen_id_table.xlsx')) ##use for determining neuroanatomical locations according to allen
    ann = sitk.ReadImage(kwargs['annotationfile'])
    if not bglst: bglst = [['Lobule VI', 936,10725,10724,10723], ['Lobule VII', 944,10728,10727,10726], ['CrI_II', 1017, 1056, 10677,10676,10675, 1064,10680,10679,10678]]


    ###Injection Site    
    sys.stdout.write('\n   processing injection site...'); sys.stdout.flush()
        
    #make outline for each structure in bglst
    sys.stdout.write('\n     generating outlines for structures of interest...'); sys.stdout.flush()
    cntlsts = []; rmlst = []
    for lst in bglst:
        nm = lst[0]        
        if cerebellum: 
            im = isolate_structures(kwargs['annotationfile'], *lst[1:])[:,390:,:]
            ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile']))[:,390:,:]
        else:
            im = isolate_structures(kwargs['annotationfile'], *lst[1:])
            ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile']))

        makedir(svlc);
        svlc2=os.path.join(svlc, nm)
        makedir(svlc2); rmlst.append(svlc2)
        impth=os.path.join(svlc2, str(kwargs['volumes'][0].brainname)+'.tif')    
        tifffile.imsave(impth, im.astype('uint8'))
        
        allen = os.path.join(svlc, 'allenatlas'); makedir(allen); rmlst.append(allen)
        if cerebellum: 
            CbAtlasFile = os.path.join(allen, 'cb_atlas.tif')   
            tifffile.imsave(CbAtlasFile, tifffile.imread(kwargs['AtlasFile'])[:,390:,:])
            cnts = allen_compare(CbAtlasFile, svlc, impth, outline = True)
        else:
            cnts = allen_compare(kwargs['AtlasFile'], svlc, impth, outline = True)
        cntlsts.append((nm, cnts))

    if not detectinjsite:        
        #load and clean up injection site
        sys.stdout.write('\n     loading injection site...'); sys.stdout.flush()
        injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
        if cerebellum: injstack = tifffile.imread(pth_update(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel, 'result.tif'))).astype('uint16')[:,390:,:]
        if not cerebellum: injstack = tifffile.imread(pth_update(os.path.join(injobj.outdr, 'elastix', injobj.brainname+'_resized_ch'+injobj.channel, 'result.tif'))).astype('uint16') 
        for i in range(len(injstack)):
            injstack[i]=cv2.medianBlur(injstack[i].astype(np.uint16), 5)
        mxx = injstack.max()
        #injstack = (injstack - injstack.min()) / (injstack.max() - injstack.min())    
        injstack[injstack<threshold] = 0; injstack=injstack*mxx
        injtif = os.path.join(svlc, 'injtif.tif'); tifffile.imsave(injtif, injstack.astype('uint16')); del injstack; rmlst.append(injtif)
    
    elif detectinjsite:
        #load detected injsite
        sys.stdout.write('\n     loading injection site...'); sys.stdout.flush()
        injobj = [xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
        lst=[v for xx in listdirfull(pth_update(injobj.injdetect3dfld)) for k,v in np.load(xx).iteritems()]
        mlst = [x for xx in lst[::2] for x in xx] #unpack slst = [x for xx in lst[1::2] for x in xx] #unpack
        try: 
            injim = np.swapaxes(np.zeros(im.shape), *kwargs['swapaxes'])
        except:
            injim = np.zeros(im.shape)
        rescale = tuple([injobj.fullsizedimensions[xx]/float(injim.shape[xx]) for xx in range(3)]) #zyx of horizontal
        for ii in mlst:
            for i in ii.plns.values():
                blnk = np.zeros(injim.shape[1:3])
                cnt = np.int32(np.asarray([[(xx[0]/rescale[2]), (xx[1]/rescale[1])] for xx in i[-1]])) #saggital
                cv2.fillPoly(blnk, [cnt], (1,1,1))
                injim[int(i[0]/rescale[0])] = blnk
        try: 
            injim = np.swapaxes(injim, *kwargs['swapaxes'])
        except:
            pass
        injtif = os.path.join(svlc, 'injtif.tif'); tifffile.imsave(injtif, injim.astype('uint16')); del injim; rmlst.append(injtif)
        
    
    #overlay contours:
    #make combined image
    sys.stdout.write('\n     making overlay...'); sys.stdout.flush()
    loc = os.path.join(svlc, 'tmp'); makedir(loc); rmlst.append(loc)
    depth.colorcode(injtif, loc)    

    ##
    grayscale=[os.path.join(allen, 'proj0.png'), os.path.join(allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]    
    color=[os.path.join(loc, 'proj0.png'), os.path.join(loc, 'proj1.png'), os.path.join(loc, 'proj2.png')]
    nametosave=[os.path.join(loc, "proj0_overlay.png"), os.path.join(loc, "proj1_overlay.png"), os.path.join(loc, "proj2_overlay.png")]        
            
    #learn to make a better colorscale
    #colorlst = [255*x for xx in np.linspace(50,255, len(bglst)) for x in pl.cm.jet(xx)[:-1]]
    #http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    colorlst = [tuple([255*x for x in colorConverter.to_rgb(xx)]) for xx in ['r', 'b', 'g', 'm', 'y', 'aqua', 'c']]
    for x in range(3):
        depth.overlay(grayscale[x], color[x], nametosave[x], alpha=0.95)
        im = cv2.imread(nametosave[x], 1)
        legend=[]        
        for xx in range(len(cntlsts)):        
            nm = cntlsts[xx][0]
            colors = colorlst[xx]
            cv2.drawContours(im, [cntlsts[xx][1][x]], 0, colors, 3)
            legend.append((nm, colors))
        if kwargs['annotationfile'][-26:-5] == 'reflectedhorizontally': im = np.flipud(im)
        cv2.imwrite(nametosave[x], im)
            
    #make overlay and add legend
    depth.layout(nametosave[0], nametosave[1], nametosave[2], svlc) ###might need 1,0,2 for some files
    im = cv2.imread(os.path.join(svlc, 'summary.png'), 1); rmlst.append(os.path.join(svlc, 'summary.png'))
    for xx in range(len(legend)):        
        nm = legend[xx][0]
        clr = legend[xx][1]
        cv2.putText(im, nm,(80,500+(xx*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr , 2)
    #brainsvlc = os.path.join(svlc, kwargs['volumes'][0].brainname); makedir(brainsvlc)
    cv2.imwrite(os.path.join(svlc, '{}_injection.png'.format(kwargs['volumes'][0].brainname)), im)
    sys.stdout.write('\n   Finished Injection Site - saved as {}\n'.format(os.path.join(svlc, 'summary.png'))); sys.stdout.flush()
    
    #remove files:
    if cleanup: [removedir(xx) for xx in rmlst]
    
    return
#%%
def get_spaced_colors(n=5):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(25, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors] 
    
    
#%% not functional
def names_of_inj_structures(im, ann, allen_id_table):
    '''Simple Function to take image stack and output list of names that are in inj site
    
    im and ann must be the same size and region (post registration)    
    '''    
    im = np.flipud(tifffile.imread(injtif))
    names=[]    
    ztick = 0
    for pln in im:
        disp, center, cnt = detect_inj_site(pln, minimumarea=20)
        try:
            print len(cnt)
            [(ztick, xx[1], xx[0]) for xx in cnt]
            transformed_pnts_to_allen_helper_func([(ztick, xx[1], xx[0]) for xx in cnt], ann)             
            ztick+=1
            break
        except:
            pass
         
         
    transformed_pnts_to_allen_helper_func(arr, ann)
             
    structure_lister(allen_id_table, *transformed_pnts_to_allen_helper_func(arr, ann))    
    iterlst=[]; [iterlst.append((arr, ann, core, cores)) for core in range(cores)]
    lst=p.map(transformed_pnts_to_allen_helper_func_par, iterlst); del iterlst
    pnt_lst=[xx for x in lst for xx in x]               

             
    return
#%%
def load_n_make_bg(path):            
    bg = pl.imread(path)
    bg = bg[...,:-1].sum(axis=-1)
    bg = np.repeat(bg[:,:,None], 4, axis=-1)
    bg = (bg-bg.min())/(bg.max()-bg.min())
    bg[:,:,-1] = 1.0
    temp=os.path.join(os.getcwd(), 'temp.png'); pl.imsave(temp, bg)
    im = cv2.imread(temp); removedir(temp)
    return im
#%%
def overlay_new(bg, mask, thresh=0.00, alpha=1):
    pl.ioff() #for spock
    # load bg
    
    bg = bg[...,:-1].sum(axis=-1)
    bg = np.repeat(bg[:,:,None], 4, axis=-1)
    bg = (bg-bg.min())/(bg.max()-bg.min())
    bg[:,:,-1] = 1.0
    
    # load mask
    
    mask[...,:-1] = (mask[...,:-1] - mask[...,:-1].min())/(mask[...,:-1].max()-mask[...,:-1].min())
    mask = zoom(mask, np.asarray(bg.shape)/np.asarray(mask.shape))

    result = bg
    idxs = np.where(mask[:,:,:-1].sum(axis=-1)>thresh)

    toadd = mask[idxs[0],idxs[1],:]
    toadd[...,-1] = alpha
    result[idxs[0],idxs[1]] *= toadd
    #pl.imsave(dest_path, result)
    #return mask,result #TP REMOVING SINCE IT SEEMS TO MESS UP SPOCK
    return result

    
def pth_update(inn):
    '''dummy function since we dont' need it and too lazy to fix'''
    return inn
    
    
    
    
    
    
