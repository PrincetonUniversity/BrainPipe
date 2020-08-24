# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:07:55 2016

@author: wanglab
"""

import os
import sys
import cv2
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pickle
import subprocess as sp
from collections import Counter
import tifffile
from tools.imageprocessing.preprocessing import resample_par
from tools.imageprocessing.preprocessing import color_movie_merger
from tools.imageprocessing.preprocessing import resample, gridcompare
from tools.imageprocessing.preprocessing import combine_images
from tools.imageprocessing import depth
from tools.objectdetection.injdetect import find_site
from tools.registration.masking import generate_masked_atlas
from tools.registration.masking import generate_cropped_atlas
from tools.utils.io import makedir, removedir, writer, load_kwargs
from tools.utils.io import convert_to_mhd


def elastix_wrapper(jobid, cores=5, **kwargs):
    '''Wrapper to handle most registration operations.

    jobid =
        0: 'normal registration'
        1: 'cellchannel inverse'
        2: 'injchannel inverse'
    '''
    # inputs
    kwargs = load_kwargs(**kwargs)
    sys.stdout.write('\nElastix in:\n')
    sys.stdout.flush()
    os.system('which elastix')

    # 'normal' registration
    if jobid == 0:
        elastix_registration(jobid, cores=cores, **kwargs)

    # cellchannel inverse
    if jobid == 1:
        make_inverse_transform(
            [xx for xx in kwargs['volumes']][0], cores=cores, **kwargs)

    # injchannel inverse -- ##FIXME think about limiting the search
    # to only the cerebellum
    if jobid == 2:
        # make inverse transform
        transformfile = make_inverse_transform(
            [xx for xx in kwargs['volumes']][0], cores=cores, **kwargs)

        # detect injection site  ##FIXME need to define image and
        # pass in appropriate thresh/filter-kernels
        inj = [xx for xx in kwargs['volumes']][0]
        # array = find_site(inj.ch_to_reg_to_atlas+'/result.1.tif',
        # thresh=10, filter_kernel=(5,5,5))

        array = find_site(inj.resampled_for_elastix_vol, thresh=10,
                          filter_kernel=(5, 5, 5)).astype(int)

        # old version
        # array = inj_detect_using_labels(threshold = .15,
        # resampledforelastix = True, num_labels_to_keep=1,
        # show = False, save = True, masking = True, **kwargs)

        # apply resizing point transform
        txtflnm = point_transform_due_to_resizing(
            array, chtype='injch', **kwargs)

        # run transformix on points
        points_file = point_transformix(txtflnm, transformfile)

        # convert registered points into structure counts
        transformed_pnts_to_allen(
            points_file, ch_type='injch', point_or_index=None, **kwargs)

    return


def elastix_registration(jobid, cores=5, **kwargs):
    '''Function to take brainvolumes and register them to
    AtlasFiles using elastix parameters in parameterfolder.
    Inputs
    ---------------
    cores = for parallelization

    optional kwargs:
    secondary_registration (optional):
        False (default) - apply transform determined from
        regch->atlas to other channels
        True - register other channel(s) to regch then apply
        transform determined from regch->atlas
        (useful if imaging conditions were different between
        channel and regch, i.e. added horizontal foci, sheet na...etc)
        kwargs overrides explicit 'secondary_registration' input to function


    Required inputs via kwargs:
            brainname='AnimalID'
            brainpath= pathtofolder
            AtlasFile=pathtoatlas ###atlas is a tif stack
            parameterfolder ##contains parameter files: Order1_filename.txt,
            Order2_filename.txt,....
            svlc=pathtosavedirectory
            maskfile(optional)=creates a nested folder inside svlc and runs
            registration with mask
    To run in parallel use: parallel_elastixlooper
  '''

    # inputs
    outdr = kwargs['outputdirectory']
    kwargs = load_kwargs(outdr)
    # check to see if masking, cropping or normal atlas
    if 'maskatlas' in kwargs:
        AtlasFile = generate_masked_atlas(**kwargs)
    elif 'cropatlas' in kwargs:
        AtlasFile = generate_cropped_atlas(**kwargs)
    else:
        AtlasFile = kwargs['AtlasFile']

    # make variables for volumes:
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]

    # images need to have been stitched, resized,
    # and saved into single tiff stack
    # resize to ~220% total size of atlas (1.3x/dim)
    sys.stdout.write('Beginning registration on {}'.format(reg_vol.brainname))
    sys.stdout.flush()
    reg_vol.add_resampled_for_elastix_vol(
        reg_vol.downsized_vol+'_resampledforelastix.tif')
    if not os.path.exists(reg_vol.resampled_for_elastix_vol):
        sys.stdout.write('\n   Resizing {}'.format(reg_vol.downsized_vol))
        sys.stdout.flush()
        resample_par(cores, reg_vol.downsized_vol+'.tif', AtlasFile,
                     svlocname=reg_vol.resampled_for_elastix_vol,
                     singletifffile=True, resamplefactor=1.3)
        [vol.add_registration_volume(
            reg_vol.resampled_for_elastix_vol) for vol in vols]
        sys.stdout.write('...completed resizing\n')
        sys.stdout.flush()

    # find elastix parameters files and sort, set up parameters and logfiles
    parameters = []
    [parameters.append(
        os.path.join(reg_vol.parameterfolder, files)) for files in os.listdir(
        reg_vol.parameterfolder) if files[0] != '.' and files[-1] != '~']
    parameters.sort()
    svlc = os.path.join(outdr, 'elastix')
    makedir(svlc)
    writer(svlc, 'Starting elastix...AtlasFile: {}\n   parameterfolder: {}\n   svlc: {}\n'.format(
        AtlasFile, reg_vol.parameterfolder, svlc))
    writer(svlc, 'Order of parameters used in Elastix:{}\n...\n\n'.format(parameters))

    # optionally generate MHD file for better scaling in elastix (make both mhds if present since one tiff and one mhd doesn't work well)
    resampled_zyx_dims = False
    if False and 'atlas_scale' in kwargs:
        atlasfilecopy = AtlasFile
        AtlasFile = convert_to_mhd(AtlasFile, dims=kwargs['atlas_scale'], dst=os.path.join(
            kwargs['outputdirectory'], os.path.splitext(os.path.basename(kwargs['AtlasFile']))[0])+'.mhd')
        # copy reg vol and calculate effective distance/pixel scale
        reg_volcopy = reg_vol.resampled_for_elastix_vol
        resampled_zyx_dims = [cc*dd for cc, dd in zip(kwargs['xyz_scale'][::-1], [float(bb) / float(
            aa) for aa, bb in zip(tifffile.imread(reg_vol.resampled_for_elastix_vol).shape, reg_vol.fullsizedimensions)])]
        # note convert_to_mhd dims are in XYZ
        reg_vol.add_resampled_for_elastix_vol(convert_to_mhd(reg_vol.resampled_for_elastix_vol, dims=resampled_zyx_dims[::-1], dst=os.path.join(
            kwargs['outputdirectory'], os.path.splitext(os.path.basename(reg_vol.resampled_for_elastix_vol))[0])+'.mhd'))

    # ELASTIX
    e_out_file, transformfile = elastix_command_line_call(
        AtlasFile, reg_vol.resampled_for_elastix_vol, svlc, parameters)

    # optionally generate MHD file for better scaling in elastix
    if False and 'atlas_scale' in kwargs:
        removedir(AtlasFile)
        removedir(AtlasFile[-3:]+'.raw')
        AtlasFile = atlasfilecopy
        removedir(reg_vol.resampled_for_elastix_vol)
        removedir(reg_vol.resampled_for_elastix_vol+'.raw')
        reg_vol.add_resampled_for_elastix_vol(reg_volcopy)

    # RG movie for visual inspection of image registration
    color_movie_merger(AtlasFile, e_out_file, svlc, reg_vol.brainname)

    # RGB movie with blue channel=pre-registered stack
    bluechannel = os.path.join(svlc, 'combinedmovies', reg_vol.brainname+'_resized_bluechannel.tif')
    # needs to be resample(not resample_par) as you need EXTACT dimensions
    resample(reg_vol.downsized_vol+'.tif', AtlasFile, svlocname=bluechannel, singletifffile=True)
    color_movie_merger(AtlasFile, e_out_file, svlc, reg_vol.brainname +
                       '_bluechanneloriginal', movie5=bluechannel)

    # Make gridline transform file
    gridfld, tmpgridline = gridcompare(svlc, reg_vol)
    sp.call(['transformix', '-in', tmpgridline, '-out', gridfld, '-tp', transformfile])
    combine_images(str(reg_vol.resampled_for_elastix_vol), AtlasFile, os.path.join(
        gridfld, 'result.tif'), e_out_file, svlc, reg_vol.brainname)
    shutil.rmtree(gridfld)

    # Apply transforms to other channels
    writer(svlc, '\n...\nStarting Transformix on channel files...\n\nChannels to process are {}\n*****\n\n'.format(
        [x.downsized_vol for x in vols]))

    # type of transform and channels to apply transform to
    secondary_registration = kwargs['secondary_registration'] if 'secondary_registration' in kwargs else True
    transform_function = apply_transformix_and_register if secondary_registration else apply_transformix
    vols_to_register = [xx for xx in vols if xx.ch_type != 'regch']

    # appy transform
    [transform_function(vol, reg_vol, svlc, cores, AtlasFile, parameters,
                        transformfile, resampled_zyx_dims) for vol in vols_to_register]
    writer(svlc, '\nPast transformix step')

    # make final output image if a cellch and injch exist
    if any([True for vol in vols_to_register if vol.ch_type == 'cellch']) and any([True for vol in vols_to_register if vol.ch_type == 'injch']):
        injch = [vol.registration_volume for vol in vols_to_register if vol.ch_type == 'injch'][0]
        cellch = [vol.registration_volume for vol in vols_to_register if vol.ch_type == 'cellch'][0]
        inj_and_cells(svlc,  cellch, injch, AtlasFile)

    # check to see if script finished due to an error
    if os.path.exists(e_out_file) == False:
        writer(
            svlc, '****ERROR****GOTTEN TO END OF SCRIPT,\nTHIS ELASTIX OUTPUT FILE DOES NOT EXIST: {0} \n'.format(e_out_file))

    # write out logfile describing parameters input into function
    writer(svlc, "\nelastixlooper has finished using:\nbrainname: {}\nAtlasFile: {}\nparameterfolder: {}\nparameter files {}\nsvlc: {}".format(
        reg_vol.brainname, AtlasFile, reg_vol.parameterfolder, parameters, svlc))

    # update volumes in kwargs and pickle
    vols_to_register.append(reg_vol)
    kwargs.update(dict([('volumes', vols_to_register)]))
    pckloc = os.path.join(outdr, 'param_dict.p')
    pckfl = open(pckloc, 'wb')
    pickle.dump(kwargs, pckfl)
    pckfl.close()
    writer(outdr, "\n\n*************STEP 3************************\nelastix has completed using:\nbrainname: {}\nAtlasFile: {}\nparameterfolder: {}\nparameter files {}\nsvlc: {}\n****************\n".format(
        reg_vol.brainname, AtlasFile, reg_vol.parameterfolder, parameters, svlc))

    return


def sp_call(call):
    print(check_output(call, shell=True))
    return


def elastix_command_line_call(fx, mv, out, parameters, fx_mask=False):
    '''Wrapper Function to call elastix using the commandline, this can be time consuming

    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' noninverse transforms)
    out = folder to save file
    parameters = list of paths to parameter files IN ORDER THEY SHOULD BE APPLIED
    fx_mask= (optional) mask path if desired

    Outputs
    --------------
    ElastixResultFile = '.tif' or '.mhd' result file
    TransformParameterFile = file storing transform parameters

    '''
    e_params = ['elastix', '-f', fx, '-m', mv, '-out', out]
    if fx_mask:
        e_params = ['elastix', '-f', fx, '-m', mv, '-fMask', fx_mask, '-out', out]

    # adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(parameters[x])
    writer(out, 'Elastix Command:\n{}\n...'.format(e_params))

    # set paths
    TransformParameterFile = os.path.join(
        out, 'TransformParameters.{}.txt'.format((len(parameters)-1)))
    ElastixResultFile = os.path.join(out, 'result.{}.tif'.format((len(parameters)-1)))

    # run elastix:
    try:
        print('Running Elastix, this can take some time....\n')
        sp.call(e_params)  # sp_call(e_params)#
        writer(out, 'Past Elastix Commandline Call')
    except RuntimeError as e:
        writer(out, '\n***RUNTIME ERROR***: {} Elastix has failed. Most likely the two images are too dissimiliar.\n'.format(e.message))
        pass
    if os.path.exists(ElastixResultFile) == True:
        writer(out, 'Elastix Registration Successfully Completed\n')
    # check to see if it was MHD instead
    elif os.path.exists(os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))) == True:
        ElastixResultFile = os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))
        writer(out, 'Elastix Registration Successfully Completed\n')
    else:
        writer(out, '\n***ERROR***Cannot find elastix result file\n: {}'.format(ElastixResultFile))
        return

    return ElastixResultFile, TransformParameterFile


def transformix_command_line_call(src, dst, transformfile):
    '''Wrapper Function to call transformix using the commandline, this can be time consuming

    Inputs
    -------------------
    src = volume path for transformation
    dst = folder to save file
    transformfile = final transform file from elastix registration

    '''
    from subprocess import check_output
    print('Running transformix, this can take some time....\n')
    # sp.call(['transformix', '-in', src, '-out', dst, '-tp', transformfile])
    call = 'transformix -in {} -out {} -tp {}'.format(src, dst, transformfile)
    print(check_output(call, shell=True))
    print('Past transformix command line Call')

    return

def transformix_plus_command_line_call(src, dst, transformfile):
    '''Wrapper Function to call transformix using the commandline, this can be time consuming

    Inputs
    -------------------
    src = volume path for transformation
    dst = folder to save file
    transformfile = final transform file from elastix registration

    '''
    from subprocess import check_output
    print('Running transformix, this can take some time....\n')
    # sp.call(['transformix', '-in', src, '-out', dst, '-tp', transformfile])
    call = 'transformix -jac all -jacmat all -def all -in {} -out {} -tp {}'.format(src, dst, transformfile)
    print(check_output(call, shell=True))
    print('Past transformix command line Call')

    return



def jacobian_command_line_call(dst, transformfile):
    '''Wrapper Function to generate jacobian DETERMINANT
    using the commandline, this can be time consuming

    Inputs
    -------------------

    dst = folder to save file
    transformfile = final transform file from elastix registration

    '''
    from subprocess import check_output
    print('Generating Jacobian determinant, this can take some time....\n')
    call = 'transformix -jac all -out {} -tp {}'.format(dst, transformfile)
    print(check_output(call, shell=True))
    print('Past Jacobian determinant command line Call')

    return


def similarity_transform(fx, mv, dst, nm, level='fast', cleanup=True):
    '''function for similarity transform

    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' n
    oninverse transforms)
    dst = location to save
    nm = 'name of file to save'
    level = 'fast', 'intermediate', 'slow' : links to parameter
    files of certain complexity
    cleanup = if False do not remove files

    Returns
    -------------
    path to file
    path to transform file (if cleanup == False)
    '''
    transform_to_use = {'slow': '/jukebox/wang/pisano/Python/lightsheet/supp_files/similarity_transform_slow.txt',
                        'intermediate': '/jukebox/wang/pisano/Python/lightsheet/supp_files/similarity_transform_intermediate.txt',
                        'fast': '/jukebox/wang/pisano/Python/lightsheet/supp_files/similarity_transform_fast.txt'}[level]

    # make dir
    makedir(dst)
    out = os.path.join(dst, 'tmp')
    makedir(out)
    fl, tp = elastix_command_line_call(fx, mv, out=out, parameters=[transform_to_use])

    #move and delete
    if nm[-4:] != '.tif':
        nm = nm+'.tif'
    dstfl = os.path.join(dst, nm)
    shutil.move(fl, dstfl)
    if cleanup:
        shutil.rmtree(out)
    print('saved as {}'.format(dstfl))

    if cleanup:
        return dstfl
    if not cleanup:
        return dstfl, tp


def apply_transformix(vol, reg_vol, svlc, cores, AtlasFile, parameters, transformfile, resampled_zyx_dims):
    '''
    Signature: (vol, svlc, cores, AtlasFile, parameters, transformfile)

    Function to
        1) apply sig/inj -> auto then registration->atlas transform
        2) generate depth coded images

    Contrast this with apply_transformix_and_register: which also includes:
    registration of a sig/inj channel to the autofluro (registration) channel

    (vol, reg_vol, svlc, cores, AtlasFile, parameters, transformfile)

    Inputs
    ----------------
    vol = volume object to apply transform
    reg_vol (unused but needed to keep input the same with
        apply_transformix_and_register)
    svlc = path to 'elastix' folder, where files will be written
    cores = int # of cores
    AtlasFile = path to ABA atlas
    parameters = list in order of application of parameter file paths
    transformfile = output of elastix's transform from reg chan to atlas

    '''

    writer(svlc, '\n\nStarting transform ONLY for: {}...\n\n   to change to transform and registration of channel to regch add "secondary_registration": True to params in run tracing'.format(vol.downsized_vol))

    # set up folder/inputs
    sig_ch_resized = vol.downsized_vol+'.tif'
    trnsfrm_outpath = os.path.join(svlc, os.path.basename(vol.downsized_vol))
    makedir(trnsfrm_outpath)
    sig_ch_resampledforelastix = sig_ch_resized[:-4]+'_resampledforelastix.tif'

    # resample if necessary
    writer(svlc, 'Resizing channel: {}'.format(sig_ch_resized))
    if not vol.ch_type == 'regch':
        resample(
            sig_ch_resized, AtlasFile, svlocname=sig_ch_resampledforelastix,
            singletifffile=True, resamplefactor=1.3)
    # cannot be resample_par because that be a pool inside of pool
    # resample_par(cores, transforminput, AtlasFile,
    # svlocname=transforminput_resized, singletifffile=True, resamplefactor=1.3)

    # optionally convert to mhd, note convert_to_mhd dims are in XYZ
    if resampled_zyx_dims:
        sig_ch_resampledforelastix = convert_to_mhd(
            vol.resampled_for_elastix_vol, dims=resampled_zyx_dims[::-1])

    # run transformix
    sp.call(['transformix', '-in', sig_ch_resampledforelastix,
             '-out', trnsfrm_outpath, '-tp', transformfile])
    writer(svlc, '\n   Transformix File Generated: {}'.format(trnsfrm_outpath))

    if resampled_zyx_dims:
        removedir(sig_ch_resampledforelastix)
        removedir(sig_ch_resampledforelastix+'.raw')

    return vol


def apply_transformix_and_register(
        vol, reg_vol, svlc, cores, AtlasFile,
        parameters, transformfile, resampled_zyx_dims):
    '''Function to
        1) register a sig/inj channel to the autofluro (registration) channel
        2) apply sig/inj -> auto then registration->atlas transform
        3) generate depth coded images

    Contrast this with apply_transformix.

    (vol, reg_vol, svlc, cores, AtlasFile, parameters, transformfile)

    Inputs
    ----------------
    vol = volume object to apply transform
    reg_vol = volume initially used to register to atlas
    svlc = path to 'elastix' folder, where files will be written
    cores = int # of cores
    AtlasFile = path to ABA atlas
    parameters = list in order of application of parameter file paths
    transformfile = output of elastix's transform from reg chan to atlas

    '''

    writer(svlc, '\n\nStarting transform AND REGISTRATION to regch for: {}...\n   to change to transform only add "secondary_registration": False to params in run tracing\n'.format(vol.downsized_vol))

    # set up folder/inputs
    sig_ch_resized = vol.downsized_vol+'.tif'
    sig_out = os.path.join(svlc, os.path.basename(vol.downsized_vol))
    makedir(sig_out)
    sig_to_reg_out = os.path.join(sig_out, 'sig_to_reg')
    makedir(sig_to_reg_out)
    reg_ch_resampledforelastix = reg_vol.resampled_for_elastix_vol
    sig_ch_resampledforelastix = sig_ch_resized[:-4]+'_resampledforelastix.tif'

    # run elastix on sig/inj channel -> reg channel
    # (but don't register reg to itself)
    if not vol.ch_type == 'regch':
        writer(svlc, 'Resizing transforminput: {}'.format(sig_ch_resized))
        resample(
            sig_ch_resized, AtlasFile, svlocname=sig_ch_resampledforelastix,
            singletifffile=True, resamplefactor=1.3)
        # cannot be resample_par because that be a pool inside of pool
        # resample_par(cores, sig_ch_resized, AtlasFile,
        # svlocname=sig_ch_resampledforelastix, singletifffile=True,
        # resamplefactor=1.3)

        ElastixResultFile, TransformParameterFile = elastix_command_line_call(
            reg_ch_resampledforelastix, sig_ch_resampledforelastix,
            sig_to_reg_out, parameters)

    # copy transform paramters to set up transform series:
    [shutil.copy(os.path.join(svlc, xx), os.path.join(sig_to_reg_out, 'regtoatlas_'+xx))
     for xx in os.listdir(svlc) if 'TransformParameters' in xx]

    # connect transforms by setting regtoatlas TP0's initial transform to sig->reg transform
    # might need to go backwards...
    reg_to_atlas_tps = [os.path.join(sig_to_reg_out, xx) for xx in os.listdir(sig_to_reg_out) if 'TransformParameters'
                        in xx and 'regtoatlas' in xx]
    reg_to_atlas_tps.sort()
    sig_to_reg_tps = [os.path.join(sig_to_reg_out, xx) for xx in os.listdir(sig_to_reg_out) if 'TransformParameters'
                      in xx and 'regtoatlas' not in xx]
    sig_to_reg_tps.sort()

    # account for moving the reg_to_atlas_tps:
    [change_transform_parameter_initial_transform(
        reg_to_atlas_tps[xx+1], reg_to_atlas_tps[xx]) for xx in range(len(reg_to_atlas_tps)-1)]

    # now make the initialtransform of the first(0) sig_to_reg be the last's reg_to_atlas transform
    change_transform_parameter_initial_transform(reg_to_atlas_tps[0], sig_to_reg_tps[-1])

    # optionally convert to mhd, note convert_to_mhd dims are in XYZ
    if resampled_zyx_dims:
        sig_ch_resampledforelastix = convert_to_mhd(
            vol.resampled_for_elastix_vol, dims=resampled_zyx_dims[::-1])

    # run transformix
    sp.call(['transformix', '-in', sig_ch_resampledforelastix,
             '-out', sig_out, '-tp', reg_to_atlas_tps[-1]])

    if resampled_zyx_dims:
        removedir(sig_ch_resampledforelastix)
        removedir(sig_ch_resampledforelastix+'.raw')

    writer(svlc, '\n   Transformix File Generated: {}'.format(sig_out))

    return vol


def change_transform_parameter_initial_transform(fl, initialtrnsformpth):
    '''
    (InitialTransformParametersFileName "NoInitialTransform")
    initialtrnsformpth = 'NoInitialTransform' or 'pth/to/transform.0.txt'
    '''
    fl1 = fl[:-5]+'0000.txt'

    with open(fl, 'r') as f, open(fl1, 'w') as new:
        for line in f:
            new.write('(InitialTransformParametersFileName "{}")\n'.format(initialtrnsformpth)
                      ) if 'InitialTransformParametersFileName' in line else new.write(line)

    # overwrite original transform file
    shutil.move(fl1, fl)
    return


def change_interpolation_order(fl, order=3):
    '''Function to change FinalBSplineInterpolationOrder of elastix file.
    This is import when pixel values need to be the same or exact

    '''
    fl1 = fl[:-5]+'0000.txt'

    with open(fl, 'r') as f, open(fl1, 'w') as new:
        for line in f:
            new.write('(FinalBSplineInterpolationOrder "{}")\n'.format(order)
                      ) if 'FinalBSplineInterpolationOrder' in line else new.write(line)

    # overwrite original transform file
    shutil.move(fl1, fl)
    return fl


def change_bspline_interpolation_order(fl, order=3):
    '''Function to change (BSplineTransformSplineOrder 3) of elastix file.
    This is import when pixel values need to be the same or exact

    '''
    fl1 = fl[:-5]+'0000.txt'

    with open(fl, 'r') as f, open(fl1, 'w') as new:
        for line in f:
            new.write('(BSplineTransformSplineOrder "{}")\n'.format(order)
                      ) if 'BSplineTransformSplineOrder' in line else new.write(line)

    # overwrite original transform file
    shutil.move(fl1, fl)
    return fl


def allen_compare(AtlasFile, svlc, trnsfrm_out_file, verbose=False, outline=False):
    '''Function to make a 3 axis depth color overlay between the allen atlas
    (will be greyscale), and the inputpth (will be color representing depth)
    AtlasFile=pth to atlas
    svlc=pth to save files
    trnsfrm_out_file=file to transform
    outline = generates an outline of the nonzero label, useful for
    interacting with allen structures rather than injeciton sites
    '''
    trnsfrm_outpath = trnsfrm_out_file[:trnsfrm_out_file.rfind('/')]
    depth.colorcode(trnsfrm_out_file, trnsfrm_outpath)
    # using Allen Atlas as background
    allen = os.path.join(svlc, 'allenatlas')
    makedir(allen)
    allentiff = AtlasFile[:-4]+'.tif'
    depth.colorcode(allentiff, allen)
    grayscale = [os.path.join(allen, 'proj0.png'), os.path.join(
        allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]
    color = [os.path.join(trnsfrm_outpath, 'proj0.png'), os.path.join(
        trnsfrm_outpath, 'proj1.png'), os.path.join(trnsfrm_outpath, 'proj2.png')]
    nametosave = [os.path.join(trnsfrm_outpath, "proj0_overlay.png"), os.path.join(
        trnsfrm_outpath, "proj1_overlay.png"), os.path.join(trnsfrm_outpath, "proj2_overlay.png")]
    ###
    if outline:
        import pylab as pl
        #cntlst = [];
        tick = 0
        for xx in color:
            im = cv2.imread(xx, 1)
            im_gray = cv2.imread(xx, 0)
            # disp, center, cnt = detect_inj_site(im_gray.astype('uint8'),
            # kernelsize = 2, threshold = 0.1, minimumarea=1);
            # cntlst.append(cnt)
            disp = find_site(im_gray)
            nim = np.zeros((im.shape[0], im.shape[1], 4))
            for i in range(3):
                if tick == 2:
                    nim[..., i] = im[..., i]*disp.astype('bool')
                if tick == 0 or tick == 1:
                    nim[..., i] = im[..., i]*disp.astype('bool')
                    nim[..., i] = nim[..., i] + im[..., i]*np.fliplr(disp.astype('bool'))
            nim[..., 3] = 1
            #cv2.imwrite(xx, nim)
            pl.imsave(xx, nim)
            tick += 1
    ##
    for x in range(3):
        depth.overlay(grayscale[x], color[x], nametosave[x], alpha=0.95)
    depth.layout(nametosave[0], nametosave[1], nametosave[2],
                 trnsfrm_outpath)  # might need 1,0,2 for some files
    if verbose != False:
        writer(svlc, "\nFinished depth coloring overlay")
    # if outline: return cntlst
    return


def inj_and_cells(elastixfld, cellch, injch, AtlasFile, threshold=0.075):
    # assumes sagittal files
    # set paths for
    # generate Allen Atlas background
    allen = os.path.join(elastixfld, 'allenatlas_inj')
    makedir(allen)
    allentiff = tifffile.imread(AtlasFile[:-4]+'.tif')[:, 390:, :]
    allentiffloc = os.path.join(allen, 'allen_for_inj.tif')
    tifffile.imsave(allentiffloc, allentiff)
    depth.colorcode(allentiffloc, allen)
    grayscale = [os.path.join(allen, 'proj0.png'), os.path.join(
        allen, 'proj1.png'), os.path.join(allen, 'proj2.png')]
    # parsing out injection site
    ninjfld = os.path.join(elastixfld, 'injection_and_cells')
    makedir(ninjfld)
    injstack = tifffile.imread(injch)[:, 390:, :]  # assumes sagittal ABA 25um
    ninjstackloc = os.path.join(ninjfld, 'injectionsite.tif')
    #normalize and threshold
    mxx = injstack.max()
    injstack = (injstack - injstack.min()) / (injstack.max() - injstack.min())
    injstack[injstack < threshold] = 0
    injstack = injstack*mxx
    tifffile.imsave(ninjstackloc, injstack.astype('uint16'))
    # applying depth coding
    depth.colorcode(ninjstackloc, ninjfld)
    # apply overlay
    color = [os.path.join(ninjfld, 'proj0.png'), os.path.join(
        ninjfld, 'proj1.png'), os.path.join(ninjfld, 'proj2.png')]
    nametosave = [os.path.join(ninjfld, "proj0_overlay.png"), os.path.join(
        ninjfld, "proj1_overlay.png"), os.path.join(ninjfld, "proj2_overlay.png")]
    for x in range(3):
        print(grayscale[x], color[x], nametosave[x])
        depth.overlay(grayscale[x], color[x], nametosave[x], alpha=0.95)
    depth.layout(nametosave[1], nametosave[0], nametosave[2], ninjfld)
    print(grayscale)
    print(color)
    print(nametosave)
    writer(elastixfld, "\nFinished inj_and_cells depth coloring overlay")
    return


def getvoxels(filelocation, savefilename=None):
    '''Function used to used to find coordinates of nonzero voxels in image. Used to quantify injection site of masked registered tiffstack
       Returns [z,y,x] coordinates
       use savefilename to instead save a numpy file of coordinates'''
    # read image and convert to np array
    image = tifffile.imread(filelocation)
    # find voxel coordinates of all points greater than zero (all should be 1)
    # note this automatically looks for voxels above 0 intensity.
    # But an arguement can be used if nonthresholded image
    voxels = np.argwhere(image)
    if savefilename == None:
        return voxels
    else:
        np.save(savefilename, voxels)
        return


def make_inverse_transform(vol_to_process, cores=5, **kwargs):
    '''Script to perform inverse transform and return path to elastix inverse parameter file

    Returns:
    ---------------
    transformfile
    '''

    sys.stdout.write('starting make_inverse_transform, this will take time...')
    # inputs
    kwargs = load_kwargs(kwargs['outputdirectory'])
    outdr = kwargs['outputdirectory']
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]
    AtlasFile = reg_vol.atlasfile
    parameterfolder = reg_vol.parameterfolder

    ###############
    ###images need to have been stitched, resized, and saved into single tiff stack ###
    ###resize to ~220% total size of atlas (1.3x/dim) ###
    reg_vol.add_resampled_for_elastix_vol(reg_vol.downsized_vol+'_resampledforelastix.tif')
    #resample_par(cores, reg_vol, AtlasFile, svlocname=reg_vol_resampled, singletifffile=True, resamplefactor=1.2)
    if not os.path.exists(reg_vol.resampled_for_elastix_vol):
        print('Resizing')
        #resample(reg_vol, AtlasFile, svlocname=reg_vol_resampled, singletifffile=True, resamplefactor=1.3)
        resample_par(cores, reg_vol.downsized_vol+'.tif', AtlasFile,
                     svlocname=reg_vol.resampled_for_elastix_vol, singletifffile=True, resamplefactor=1.3)
        print('Past Resizing')

    vol_to_process.add_resampled_for_elastix_vol(
        vol_to_process.downsized_vol+'_resampledforelastix.tif')

    if not os.path.exists(vol_to_process.resampled_for_elastix_vol):
        print('Resizing')
        resample_par(cores, vol_to_process.downsized_vol+'.tif', AtlasFile,
                     svlocname=vol_to_process.resampled_for_elastix_vol, singletifffile=True, resamplefactor=1.3)
        print('Past Resizing')

    # setup
    parameters = []
    [parameters.append(os.path.join(parameterfolder, files))
     for files in os.listdir(parameterfolder) if files[0] != '.' and files[-1] != '~']
    parameters.sort()

    # set up save locations
    svlc = os.path.join(outdr, 'elastix_inverse_transform')
    makedir(svlc)
    svlc = os.path.join(svlc, '{}_{}'.format(vol_to_process.ch_type, vol_to_process.brainname))
    makedir(svlc)

    # Creating LogFile
    #writer(svlc, 'Starting elastix...AtlasFile: {}\n   parameterfolder: {}\n   svlc: {}\n'.format(AtlasFile, parameterfolder, svlc))
    writer(svlc, 'Order of parameters used in Elastix:{}\n...\n\n'.format(parameters))

    # register: 1) atlas->reg 2) reg->sig NOTE these are intentionally backwards so applying point transform can be accomplished
    # atlas(mv)->reg (fx)
    atlas2reg = os.path.join(
        svlc, reg_vol.resampled_for_elastix_vol[reg_vol.resampled_for_elastix_vol.rfind('/')+1:-4]+'_atlas2reg')
    makedir(atlas2reg)
    e_out_file, e_transform_file = elastix_command_line_call(
        fx=reg_vol.resampled_for_elastix_vol, mv=AtlasFile, out=atlas2reg, parameters=parameters)

    # reg(mv)->sig(fx)
    reg2sig = os.path.join(
        svlc, vol_to_process.resampled_for_elastix_vol[vol_to_process.resampled_for_elastix_vol.rfind('/')+1:-4]+'_reg2sig')
    makedir(reg2sig)
    e_out_file, e_transform_file = elastix_command_line_call(
        fx=vol_to_process.resampled_for_elastix_vol, mv=reg_vol.resampled_for_elastix_vol, out=reg2sig, parameters=parameters)

    # set up transform series:
    atlas2reg2sig = os.path.join(
        svlc, vol_to_process.resampled_for_elastix_vol[vol_to_process.resampled_for_elastix_vol.rfind('/')+1:-4]+'_atlas2reg2sig')
    makedir(atlas2reg2sig)
    # copy transform paramters
    [shutil.copy(os.path.join(reg2sig, xx), os.path.join(atlas2reg2sig, 'reg2sig_'+xx))
     for xx in os.listdir(reg2sig) if 'TransformParameters' in xx]
    [shutil.copy(os.path.join(atlas2reg, xx), os.path.join(atlas2reg2sig, 'atlas2reg_'+xx))
     for xx in os.listdir(atlas2reg) if 'TransformParameters' in xx]

    # connect transforms by setting regtoatlas TP0's initial transform to sig->reg transform
    tps = [os.path.join(atlas2reg2sig, xx)
           for xx in os.listdir(atlas2reg2sig) if 'TransformParameters' in xx]
    # they are now in order recent to first, thus first is regtoatlas_TransformParameters.1.txt
    tps.sort(reverse=True)
    for x in range(len(tps)):
        if not x == len(tps)-1:
            change_transform_parameter_initial_transform(tps[x], tps[x+1])

    assert os.path.exists(tps[0])
    writer(svlc, '***Elastix Registration Successfully Completed***\n')
    writer(svlc, '\ne_transform_file is {}'.format(tps[0]))
    ####################
    sys.stdout.write('complted make_inverse_transform')
    return tps[0]

    ############################################################################################################
    ######################apply point transform and make transformix input file#################################
    ############################################################################################################
    # find centers and add 1's to make nx4 array for affine matrix multiplication to account for downsizing
    # everything is in PIXELS


def point_transform_due_to_resizing(array, chtype='injch', svlc=False, **kwargs):
    '''Function to take npy array, find nonzero pixels, apply point transform (due to resizing) and package them into a file for final elastix point transform


    Inputs
    -------------
    array = np array, tif, or path to numpy array from tools.objectdetection.injdetect.inj_detect_using_labels ZYX****
    chtype = 'injch' or 'cellch'
    svlc =
        False: savesfile into (outdr, 'transformedpoints_pretransformix'). Strongly recommended to use this as it will then work with the rest of package
        True

    Returns
    ---------------
    txtflnm = pth to file containing transformed points BEFORE elastix transformation

    NOTE THIS FUNCTION ASSUMES ARRAY AND ATLAS HAVE SAME Z,Y,X (DOES NOT TAKE INTO ACCOUNT SWAPPING OF AXES)
    '''
    if type(array) == str:
        if array[-4:] == '.npy':
            array = np.load(array)
        if array[-4:] == '.tif':
            array = tifffile.imread(array)

    kwargs = load_kwargs(**kwargs)
    outdr = kwargs['outputdirectory']
    brainname = [xx for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0].brainname
    vol = [xx for xx in kwargs['volumes'] if xx.ch_type == chtype][0]

    # array dimensions
    z, y, x = array.shape

    # compare size of array with 'resampledforelastixsize'
    with tifffile.TiffFile([os.path.join(outdr, f) for f in os.listdir(outdr) if 'resampledforelastix.tif' in f and not '3D_contours' in f and 'ch{}'.format([xx.channel for xx in kwargs['volumes'] if xx.ch_type == 'regch'][0]) in f][0]) as tif:
        zr = len(tif.pages)
        yr, xr = tif.pages[0].shape
        tif.close()

    nonzeropixels = np.argwhere(array > 0)
    nx4centers = np.ones((len(nonzeropixels), 4))  # FIXME: the logic needs to be checked
    nx4centers[:, :-1] = nonzeropixels

    # create transformmatrix
    trnsfrmmatrix = np.identity(4)*(zr/z, yr/y, xr/x, 1)  # downscale to "resampledforelastix size"
    sys.stdout.write('\n\n Transfrom matrix:\n{}\n'.format(trnsfrmmatrix))

    # nx4 * 4x4 to give transform
    trnsfmdpnts = nx4centers.dot(trnsfrmmatrix)  # z,y,x
    sys.stdout.write('\nfirst three transformed pnts:\n{}\n'.format(trnsfmdpnts[0:3]))

    # create txt file, with elastix header, then populate points
    txtflnm = '{}_zyx_transformedpnts_{}.txt'.format(brainname, vol.ch_type)

    #
    if not svlc:
        pnts_fld = os.path.join(outdr, 'transformedpoints_pretransformix')
        makedir(pnts_fld)
    if svlc:
        pnts_fld = os.path.join(svlc)
        makedir(pnts_fld)

    transforminput = os.path.join(pnts_fld, txtflnm)
    removedir(transforminput)  # prevent adding to an already made file
    writer(pnts_fld, 'index\n{}\n'.format(len(trnsfmdpnts)), flnm=txtflnm)
    sys.stdout.write('\nwriting centers to transfomix input points text file...')
    stringtowrite = '\n'.join(['\n'.join(['{} {} {}'.format(i[2], i[1], i[0])])
                               for i in trnsfmdpnts])  # this step converts from zyx to xyz*****
    writer(pnts_fld, stringtowrite, flnm=txtflnm)
    # [writer(pnts_fld, '{} {} {}\n'.format(i[2],i[1],i[0]), flnm=txtflnm, verbose=False) for i in trnsfmdpnts] ####this step converts from zyx to xyz*****
    sys.stdout.write('...done writing centers.\nSaved in {}'.format(pnts_fld))
    sys.stdout.flush()

    del trnsfmdpnts, trnsfrmmatrix, nx4centers

    return os.path.join(pnts_fld, txtflnm)


def point_transformix(txtflnm, transformfile, dst=False):
    '''apply elastix transform to points


    Inputs
    -------------
    txtflnm = list of points that have resizingtransform
    transformfile = elastix transform file
    dst = optional folder

    Returns
    ---------------
    trnsfrm_out_file = pth to file containing post transformix points

    '''
    sys.stdout.write('\n***********Starting Transformix***********')
    from subprocess import check_output
    # set paths
    if not dst:
        trnsfrm_outpath = os.path.join(os.path.dirname(transformfile), 'posttransformix')
        makedir(trnsfrm_outpath)
    if dst:
        trnsfrm_outpath = os.path.join(dst, 'posttransformix')
        makedir(trnsfrm_outpath)
    trnsfrm_out_file = os.path.join(trnsfrm_outpath, 'outputpoints.txt')

    # run transformix point transform
    call = 'transformix -def {} -out {} -tp {}'.format(txtflnm, trnsfrm_outpath, transformfile)
    print(check_output(call, shell=True))
    writer(trnsfrm_outpath, '\n   Transformix File Generated: {}'.format(trnsfrm_out_file))
    return trnsfrm_out_file


def collect_points_post_transformix(src, point_or_index='point'):
    '''
    src = output text file from point transformix

    returns array XYZ
    '''

    idx = 'OutputPoint' if point_or_index == 'point' else 'OutputIndexFixed'

    with open(src, 'r') as fl:
        lines = fl.readlines()
        fl.close()

    # populate post-transformed array of contour centers
    arr = np.empty((len(lines), 3))
    for i in range(len(lines)):
        arr[i, ...] = lines[i].split()[lines[i].split().index(
            idx)+3:lines[i].split().index(idx)+6]  # x,y,z

    return arr


def transformed_pnts_to_allen(points_file, ch_type='injch', point_or_index=None, allen_id_table_pth=False, **kwargs):
    '''function to take elastix point transform file and return anatomical locations of those points
    point_or_index=None/point/index: determines which transformix output to use: point is more accurate, index is pixel value(?)
    Elastix uses the xyz convention rather than the zyx numpy convention

    Inputs
    -----------
    points_file =
    ch_type = 'injch' or 'cellch'
    allen_id_table_pth (optional) pth to allen_id_table

    Returns
    -----------
    excelfl = path to excel file

    '''
    kwargs = load_kwargs(**kwargs)
    # inputs
    assert type(points_file) == str

    if point_or_index == None:
        point_or_index = 'OutputPoint'
    elif point_or_index == 'point':
        point_or_index = 'OutputPoint'
    elif point_or_index == 'index':
        point_or_index = 'OutputIndexFixed'

    #
    vols = kwargs['volumes']
    reg_vol = [xx for xx in vols if xx.ch_type == 'regch'][0]

    # load files
    if not allen_id_table_pth:
        # use for determining neuroanatomical locations according to allen
        allen_id_table = pd.read_excel(os.path.join(
            reg_vol.packagedirectory, 'supp_files/allen_id_table.xlsx'))
    else:
        allen_id_table = pd.read_excel(allen_id_table_pth)
    ann = sitk.GetArrayFromImage(sitk.ReadImage(kwargs['annotationfile']))  # zyx
    with open(points_file, "rb") as f:
        lines = f.readlines()
        f.close()

    # populate post-transformed array of contour centers
    sys.stdout.write('\n{} points detected\n\n'.format(len(lines)))
    arr = np.empty((len(lines), 3))
    for i in range(len(lines)):
        arr[i, ...] = lines[i].split()[lines[i].split().index(point_or_index) +
                                       3:lines[i].split().index(point_or_index)+6]  # x,y,z

    # optional save out of points
    np.save(kwargs['outputdirectory']+'/injection/zyx_voxels.npy',
            np.asarray([(z, y, x) for x, y, z in arr]))

    pnts = transformed_pnts_to_allen_helper_func(arr, ann)
    pnt_lst = [xx for xx in pnts if xx != 0]

    # check to see if any points where found
    if len(pnt_lst) == 0:
        raise ValueError('pnt_lst is empty')
    else:
        sys.stdout.write('\nlen of pnt_lst({})\n\n'.format(len(pnt_lst)))

    # generate dataframe with column
    df = count_structure_lister(allen_id_table, *pnt_lst)

    # save df
    nametosave = '{}_{}'.format(reg_vol.brainname, ch_type)
    excelfl = os.path.join(kwargs['outputdirectory'], nametosave + '_stuctures_table.xlsx')
    df.to_excel(excelfl)
    print('\n\nfile saved as: {}'.format(excelfl))

    return excelfl


def transformed_pnts_to_allen_helper_func(arr, ann, order='XYZ'):
    '''Function to transform given array of indices and return the atlas pixel ID from the annotation file

    Input
    --------------
    numpy array of Nx3 dimensions corresponding to ***XYZ*** coordinates
    ann = numpy array of annotation file
    order = 'XYZ' or 'ZYX' representing the dimension order of arr's input

    Returns
    -------------
    Pixel value at those indices of the annotation file
    '''
    # procecss
    pnt_lst = []
    badpntlst = []
    for i in range(len(arr)):
        try:
            pnt = [int(x) for x in arr[i]]
            if order == 'XYZ':
                pnt_lst.append(ann[pnt[2], pnt[1], pnt[0]])  # find pixel id; arr=XYZ; ann=ZYX
            elif order == 'ZYX':
                pnt_lst.append(ann[pnt[0], pnt[1], pnt[2]])  # find pixel id; arr=ZYX; ann=ZYX
        except IndexError:
            badpntlst.append([pnt[2], pnt[1], pnt[0]])  # ZYX
            pass  # THIS NEEDS TO BE CHECKED BUT I BELIEVE INDEXES WILL BE OUT OF
    sys.stdout.write(
        '\n*************{} points do not map to atlas*********\n'.format(len(badpntlst)))
    sys.stdout.flush()
    return pnt_lst


def count_structure_lister(allen_id_table, *args):
    '''Function that generates a pd table of structures where contour detection has been observed
    Inputs:
        allen_id_table=annotation file as np.array
        *args=list of allen ID pixel values ZYX
    '''
    # make dictionary of pixel id:#num of the id
    cnt = Counter()
    for i in args:
        cnt[i] += 1

    # generate df + empty column
    if type(allen_id_table) == str:
        # df=allen_id_table.assign(count= [0]*len(allen_id_table)) #add count columns to df
        allen_id_table = pd.read_excel(allen_id_table)
    df = allen_id_table
    df['cell_count'] = 0

    # populate cell count in dataframe
    for pix_id, count in cnt.items():
        df.loc[df.id == pix_id, 'cell_count'] = count

    return df
