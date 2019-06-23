# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tools.registration.register import *

pths = ['/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171012_201707_tp06', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171012_201707_mk27', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171012_201707_mk23', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171012_201707_mk22', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_tp05', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_tp04', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_tp02', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk65', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk63', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk62', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk61', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk07', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk05', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171011_201707_mk01', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170818_201707_tp07', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170818_201707_mk06', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170818_201707_mk04', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170818_201707_mk03', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170817_201707_mk66', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170817_201707_mk02_redone', 
'/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/170817_201707_mk02']

for pth in pths:
    kwargs = load_kwargs(pth)
    reg_vol=[xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0]
    if pth != '/home/tpisano/wang/mkislin/lightsheet_brains/201707_ofp_mini/171012_201707_tp06':
        try:
            shutil.move(reg_vol.elastixfld, reg_vol.elastixfld+'_old')
        except IOError, e:
                print e
    #removedir(reg_vol.elastixfld)
    #removedir(reg_vol.inverse_elastixfld)
    elastix_wrapper(0, cores=10, **kwargs) #only doing normal transform, use 1 or 2 for inverse
    
#%%

from tools.registration.register import *
from tools.imageprocessing.preprocessing import *
pths = ['/home/wanglab/wang/pisano/tracing_output/antero/20170419_db_bl6_cri_rpv_53h',
        '/home/wanglab/wang/pisano/tracing_output/antero/20170419_db_cri_mid_53h',
        '/home/wanglab/wang/pisano/tracing_output/antero/20170204_tp_bl6_cri_250r_01',
        '/home/wanglab/wang/pisano/tracing_output/antero/20170204_tp_bl6_cri_1750r_03',
        '/home/wanglab/wang/pisano/tracing_output/antero/20170204_tp_bl6_cri_1000r_02']

badpaths = []
for pth in pths:
    try:
        kwargs = load_kwargs(pth)
        updateparams(cwd=False, svnm=False, **kwargs)
        reg_vol=[xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0]
        inj_vol=[xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
        
        ###inputs
        outdr = kwargs['outputdirectory']
        kwargs = load_kwargs(outdr)
        #check to see if masking, cropping or normal atlas
        if 'maskatlas' in kwargs:
            AtlasFile = generate_masked_atlas(**kwargs)
        elif 'cropatlas' in kwargs:
            AtlasFile = generate_cropped_atlas(**kwargs)  
        else: 
            AtlasFile=kwargs['AtlasFile']
            
        ###make variables for volumes:
        vols=kwargs['volumes']
        reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]
        
        transformfile = reg_vol.elastixfld+'/TransformParameters.1.txt'        
                    
        parameters=[]; [parameters.append(os.path.join(reg_vol.parameterfolder, files)) for files in os.listdir(reg_vol.parameterfolder) if files[0] != '.' and files [-1] != '~']; parameters.sort()       
        svlc=os.path.join(outdr, 'elastix'); makedir(svlc)
        
        secondary_registration = kwargs['secondary_registration'] if 'secondary_registration' in kwargs else False
        transform_function = apply_transformix_and_register if secondary_registration else apply_transformix
        vols_to_register=[xx for xx in vols if xx.ch_type != 'regch']   
        [transform_function(vol, reg_vol, svlc, 2, AtlasFile, parameters, transformfile) for vol in vols_to_register]                  
    except:
        badpaths.append(pth)
    
    print 'Done badpaths:'
    print badpaths
    
#%%
import shutil
from tools.utils.io import load_kwargs
from tools.utils.update import update_lightsheet_folder
from tools.registration.register import *
from tools.imageprocessing.preprocessing import *    
pths = ['/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W17_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W19_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W21_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W22_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W23_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_mli/DREADD_3W102_YFP', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W6_YFP_Pcp2', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W7_YFP_Pcp2', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W9_YFP_Pcp2', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W21_YFP_Pcp2', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W24_YFP_Pcp2', '/jukebox/wang/Jess/lightsheet_output/dreadds_pc/DREADD_3W35_YFP_Pcp2']


badpaths = []
for pth in pths:
    try:
        update_lightsheet_folder(pth, updateruntracing=False)
        kwargs = load_kwargs(pth)
        updateparams(cwd=False, svnm=False, **kwargs)
        reg_vol=[xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0]
        inj_vol=[xx for xx in kwargs['volumes'] if xx.ch_type == 'injch'][0]
        
        ###inputs
        outdr = kwargs['outputdirectory']
        kwargs = load_kwargs(outdr)
        #check to see if masking, cropping or normal atlas
        if 'maskatlas' in kwargs:
            AtlasFile = generate_masked_atlas(**kwargs)
        elif 'cropatlas' in kwargs:
            AtlasFile = generate_cropped_atlas(**kwargs)  
        else: 
            AtlasFile=kwargs['AtlasFile']
            
        ###make variables for volumes:
        vols=kwargs['volumes']
        reg_vol=[xx for xx in vols if xx.ch_type == 'regch'][0]
        
        transformfile = reg_vol.elastixfld+'/TransformParameters.1.txt'        
                    
        parameters=[]; [parameters.append(os.path.join(reg_vol.parameterfolder, files)) for files in os.listdir(reg_vol.parameterfolder) if files[0] != '.' and files [-1] != '~']; parameters.sort()       
        svlc=os.path.join(outdr, 'elastix'); makedir(svlc)
        resampled_zyx_dims = [cc*dd for cc, dd in zip(kwargs['xyz_scale'][::-1], [float(bb) / float(aa) for aa, bb in zip(tifffile.imread(reg_vol.resampled_for_elastix_vol).shape, reg_vol.fullsizedimensions)])]
        secondary_registration = kwargs['secondary_registration'] if 'secondary_registration' in kwargs else False
        transform_function = apply_transformix_and_register if secondary_registration else apply_transformix
        vols_to_register=[xx for xx in vols if xx.ch_type != 'regch']
        
        ###CLEAN###
        [shutil.rmtree(os.path.dirname(vol.ch_to_reg_to_atlas)) for vol in vols_to_register]
        
        [transform_function(vol, reg_vol, svlc, 2, AtlasFile, parameters, transformfile, resampled_zyx_dims) for vol in vols_to_register] 
    except:
        badpaths.append(pth)
    
    print 'Done badpaths:'
    print badpaths
    
    
