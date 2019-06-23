#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:32:27 2017

@author: tpisano
"""
if False:
    from tools.utils.setup_multi import setup_list
    import os, shutil
    from tools.utils.update import change_line_in_text_file
    from tools.imageprocessing.preprocessing import generateparamdict
    run_tracing_file = os.path.join(os.getcwd(), 'run_tracing.py') 
    
    lst = [  '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180202_20171130_pcp2_dcn_01_488_647_015na_1hfds_z5um_150msec_15-23-11',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180202_20171130_pcp2_dcn_01_790_015na_1hfds_z5um_1000msec_17-44-41',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_02_488_647_015na_1hfds_z5um_150msec_12-31-17',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_02_790_015na_1hfds_z5um_1000msec_11-27-49',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_03_488_647_015na_1hfds_z5um_150msec_15-10-04',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_03_790_015na_1hfds_z5um_1000msec_14-10-22',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_04_488_647_015na_1hfds_z5um_150msec_18-36-08',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180205_20171130_pcp2_dcn_04_790_015na_1hfds_z5um_1000msec_17-22-18',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_05_488_647_015na_1hfds_z5um_150msec_11-13-59',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_05_790_015na_1hfds_z5um_1000msec_10-14-28',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_06_488_647_015na_1hfds_z5um_150msec_13-33-31',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_06_790_015na_1hfds_z5um_1000msec_12-26-56',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_07_488_647_015na_1hfds_z5um_150msec_16-06-53',
             '/home/wanglab/LightSheetTransfer/tp/pcp2_dcn/180206_20171130_pcp2_dcn_07_790_015na_1hfds_z5um_1000msec_14-32-22']
        

    for xx in range(0, len(lst), 2):
        
        #BE SURE TO CHECK LINE VARIABLES EACH TIME
        autoline = "os.path.join(systemdirectory, 'LightSheetTransfer"
        auto = "os.path.join(systemdirectory, '{}'): [['regch', '00'], ['injch', '01']],os.path.join(systemdirectory, '{}'): [['injch', '00']]}}".format(lst[xx][14:], lst[xx+1][14:])
    
        outdirline = "'outputdirectory': os.path.join(systemdirectory,"
        outdir = "'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/tracing_output/aav/{}'),".format(lst[xx][52:72])

        outpath = os.path.join('/jukebox/wang/pisano/tracing_output/aav/{}'.format(lst[xx][52:72]))
        print auto, outdir, outpath
        print
        #change parts of run_tracing.py file
        change_line_in_text_file(run_tracing_file, original_text = autoline, new_text = auto)
        change_line_in_text_file(run_tracing_file, original_text = outdirline, new_text = outdir)
        #then copy out
        if not os.path.exists(os.path.join(outpath, 'lightsheet')): shutil.copytree('/jukebox/wang/pisano/Python/lightsheet', os.path.join(outpath, 'lightsheet'), ignore=shutil.ignore_patterns('^.git')) #copy run folder into output to save run info
    

#%%
if False:
    from tools.utils.setup_multi import setup_list
    import os, shutil
    from tools.utils.update import change_line_in_text_file
    from tools.imageprocessing.preprocessing import generateparamdict
    run_tracing_file = os.path.join(os.getcwd(), 'run_tracing.py') 
    
    lst = ['/home/wanglab/LightSheetTransfer/tp/ymaze2/180131_20171129_ymaze_cfos_16_488_555_015na_1hfds_z5um_150msec_11-23-41',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180131_20171129_ymaze_cfos_17_488_555_015na_1hfds_z5um_150msec_14-25-48',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180131_20171129_ymaze_cfos_19_488_555_015na_1hfds_z5um_150msec_17-45-49',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180201_20171129_ymaze_cfos_18_488_555_015na_1hfds_z5um_150msec_11-08-20',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180201_20171129_ymaze_cfos_20_488_555_015na_1hfds_z5um_150msec_19-47-45',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180201_20171129_ymaze_cfos_21_488_555_015na_1hfds_z5um_150msec_13-15-21',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180201_20171129_ymaze_cfos_22_488_555_015na_1hfds_z5um_150msec_15-18-47',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180201_20171129_ymaze_cfos_23_488_555_015na_1hfds_z5um_150msec_17-24-34',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180202_20171129_ymaze_cfos_24_488_555_015na_1hfds_z5um_150msec_19-28-53',
           '/home/wanglab/LightSheetTransfer/tp/ymaze2/180202_20171129_ymaze_cfos_25_488_555_015na_1hfds_z5um_150msec_12-48-11']


    for xx in range(0, len(lst)):
        
        #BE SURE TO CHECK LINE VARIABLES EACH TIME
        autoline = "os.path.join(systemdirectory, 'LightSheetTransfer"
        auto = "os.path.join(systemdirectory, '{}'): [['regch', '00'], ['injch', '01']]}}".format(lst[xx][14:])
    
        outdirline = "'outputdirectory': os.path.join(systemdirectory,"
        outdir = "'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/ymaze/lightsheet_analysis/injection/{}'),".format(lst[xx][50:69]+lst[xx][70:72])

        outpath = os.path.join('/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection/{}'.format(lst[xx][50:69]+lst[xx][70:72]))
        print auto, outdir, outpath
        print
        #change parts of run_tracing.py file
        change_line_in_text_file(run_tracing_file, original_text = autoline, new_text = auto)
        change_line_in_text_file(run_tracing_file, original_text = outdirline, new_text = outdir)
        #then copy out
        if not os.path.exists(os.path.join(outpath, 'lightsheet')): shutil.copytree('/jukebox/wang/pisano/Python/lightsheet', os.path.join(outpath, 'lightsheet'), ignore=shutil.ignore_patterns('^.git')) #copy run folder into output to save run info
    
#%%


#unfinished
if __name__ == False:
    from tools.utils.setup_multi import setup_list
    from tools.utils.update import change_line_in_text_file
    from tools.imageprocessing.preprocessing import generateparamdict
    run_tracing_file = os.path.join(os.getcwd(), 'run_tracing.py') 

    lst = ['/home/wanglab/LightSheetTransfer/tp/170926_20170611_bl6_prv_01_010na_488_555_647_790_1hfsds_z5um_300msec_14-33-53', '/home/wanglab/LightSheetTransfer/tp/170927_20170611_bl6_prv_03_010na_488_555_647_790_1hfsds_z5um_300msec_09-17-37', '/home/wanglab/LightSheetTransfer/tp/170927_20170611_bl6_prv_08_010na_488_555_647_790_1hfsds_z5um_300msec_11-14-19', '/home/wanglab/LightSheetTransfer/tp/171002_20170611_bl6_prv_10_010na_488_555_647_1hfsds_z5um_300msec_14-19-25', '/home/wanglab/LightSheetTransfer/tp/171002_20170611_bl6_prv_15_010na_488_555_647_1hfsds_z5um_300msec_16-54-26', '/home/wanglab/LightSheetTransfer/tp/171002_20170611_bl6_prv_05_010na_488_555_647_1hfsds_z5um_300msec_19-28-16', '/home/wanglab/LightSheetTransfer/tp/171009_20170611_bl6_prv_14_010na_488_555_647_1hfsds_z5um_300msec_12-19-57', '/home/wanglab/LightSheetTransfer/tp/171009_20170611_bl6_prv_09_010na_488_555_647_1hfsds_z5um_300msec_13-57-21', '/home/wanglab/LightSheetTransfer/tp/171009_20170611_bl6_prv_06_010na_488_555_647_1hfsds_z5um_300msec_16-03-14', '/home/wanglab/LightSheetTransfer/tp/171018_20170611_bl6_prv_07_010na_488_555_647_1hfsds_z5um_300msec_08-49-30', '/home/wanglab/LightSheetTransfer/tp/171018_20170611_bl6_prv_11_010na_488_555_647_1hfsds_z5um_300msec_12-12-41', '/home/wanglab/LightSheetTransfer/tp/171018_20170611_bl6_prv_02_010na_488_555_647_1hfsds_z5um_300msec_17-19-25', '/home/wanglab/LightSheetTransfer/tp/171019_20170611_bl6_prv_04_010na_488_555_647_1hfsds_z5um_300msec_20-58-18']


    for xx in range(0, len(lst)):
        autoline = "os.path.join(systemdirectory, 'LightSheetTransf"
        auto = "os.path.join(systemdirectory, '{}'): [['regch', '00'], ['injch', '01'], ['cellch', '01'], ['cellch', '02']]}}".format(lst[xx][14:])
    
        outdirline = "'outputdirectory': os.path.join(systemdirectory, 'wan"
        outdir = "'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/tracing_output/prv/{}'),".format(lst[xx][43:62])

        outpath = os.path.join('/home/wanglab/wang/pisano/tracing_output/prv/{}'.format(lst[xx][43:62]))
        print outpath
        #change parts of run_tracing.py file
        change_line_in_text_file(run_tracing_file, original_text = autoline, new_text = auto)
        change_line_in_text_file(run_tracing_file, original_text = outdirline, new_text = outdir)
        #gen params
        from run_tracing import params
        generateparamdict(**params)
        del params
        #then copy out
        if not os.path.exists(os.path.join(outpath, 'lightsheet')): shutil.copytree('/home/wanglab/wang/pisano/Python/lightsheet', os.path.join(outpath, 'lightsheet'), ignore=shutil.ignore_patterns('^.git')) #copy run folder into output to save run info
    
    
    
#%%
if __name__ == False:
    lst = ['/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos02', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos03', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos04', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos05', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos06', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos07', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos08', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos09', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos10', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos11', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos12', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos13', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos14', '/jukebox/wang/pisano/ymaze/lightsheet_analysis/injection_aba/20170915_ymaze_cfos15']
    #sbatch: error: Batch job submission failed: I/O error writing script/environment to file
    import os, shutil
    badpaths=[]
    for pth in lst:
        print pth
        os.chdir(pth+'/lightsheet')
        from tools.utils.process_local import run_brain_locally
        try:
            run_brain_locally(pth, cores = 12, steps=[0,1,2,3])
            shutil.rmtree(os.path.join(pth, 'full_sizedatafld'))
        except Exception, e:
            print ('Moving along, despite error: {}'.format(e))
            badpaths.append(pth)
        print badpaths
    





#%%    
def setup_file(lst, **kwargs):
    '''
    inputline = "os.path.join(systemdirectory, '{}'): [['regch', '00'], ['injch', '01'], ['cellch', '01'], ['cellch', '02']]}}".format(lst[xx][14:])
    '''
    
    
    input_original = kwargs['in_original'] if 'in_original' in kwargs else "os.path.join(systemdirectory, 'LightSheetTransf"
    input_range = kwargs['input_range'] if 'input_range' in kwargs else '14:'
    input_change = kwargs['input_change'] if 'input_change' in kwargs else "os.path.join(systemdirectory, '{}'): [['regch', '00'], ['injch', '01'], ['cellch', '01'], ['cellch', '02']]}}".format(lst[xx][eval(input_range)])
    
    for xx in range(0, len(lst)):
        line = eval('lst[xx][{}]'.format(input_range))
        input_change = eval('{}'.format(input_change))
        
    
        outdirline = "'outputdirectory': os.path.join(systemdirectory, 'wan"
        outdir = "'outputdirectory': os.path.join(systemdirectory, 'wang/pisano/tracing_output/prv/{}'),".format(lst[xx][43:62])

        outpath = os.path.join('/home/wanglab/wang/pisano/tracing_output/prv/{}'.format(lst[xx][43:62]))
        print outpath
        #change parts of run_tracing.py file
        change_line_in_text_file(run_tracing_file, original_text = in_original, new_text = input_change)
        change_line_in_text_file(run_tracing_file, original_text = outdirline, new_text = outdir)
        #then copy out
        if not os.path.exists(os.path.join(outpath, 'lightsheet')): shutil.copytree('/home/wanglab/wang/pisano/Python/lightsheet', os.path.join(outpath, 'lightsheet'), ignore=shutil.ignore_patterns('^.git')) #copy run folder into output to save run info
    
    
    return
    
    
 #%% 
    
def replace_lines(filepath, linenumber_text_list, verbose = False):
    '''Function to replace string of text of filepath.
    
    File must exist previously. 
    Line number is ****1 based numberics****
    
    Inputs:
    -------------
    filepath: local path to clearmap_cluster package
    linenumber_textr_list: list of line number in lightsheet file, and text to write in that line. e.g. lst = [[22, 'I want this string to go on line 22'], [23, 'this string on 23']]
    
    ''' 
    if verbose: sys.stdout.write('\nOpening file {}\n\n'.format(filepath))
    
    #open file
    with open(filepath, 'r') as fl:
        lines = fl.readlines()
        fl.close()

    #replace each line
    for xx in linenumber_text_list:
        lines[xx[0]-1] = xx[1]+'\n'
        if verbose: sys.stdout.write('\nReplacing line {}:\n   {}'.format(xx[0], xx[1]))
        
    #rewrite file
    with open(filepath, 'w') as fl:
        fl.writelines(lines)
        fl.close()

    if verbose: sys.stdout.write('\nRewriting file as {}\n\n'.format(filepath))

    return

def setup_list(run_tracing_file, *args):
    '''Function to set up multiple clearmap_cluster folders.
    Must be run from within the run_clearmap_cluster package.

    Inputs
    ---------------
    run_tracing_file : full path to run_tracing_file.py file
    *args = list of 
    '''

    #modify run_lightsheet_file lst
    [replace_lines(run_tracing_file, xx, verbose = True) for xx in args]

    return
