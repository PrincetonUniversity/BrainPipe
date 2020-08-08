#! /bin/env python

import os, sys
import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch
import SimpleITK as sitk


import logging
import argparse
import time
import pickle
import json

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

def make_info_file(volume_size,resolution,layer_dir,commit=True,atlas_type=None):
    """ 
    ---PURPOSE---
    Make the cloudvolume info file.
    ---INPUT---
    volume_size     [Nx,Ny,Nz] in voxels, e.g. [2160,2560,1271]
    pix_scale_nm    [size of x pix in nm,size of y pix in nm,size of z pix in nm], e.g. [5000,5000,10000]
    commit          if True, will write the info/provenance file to disk. 
                    if False, just creates it in memory
    """
    info = CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = 'image', # 'image' or 'segmentation'
        data_type = 'uint16', # 
        encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution = resolution, # Size of X,Y,Z pixels in nanometers, 
        voxel_offset = [ 0, 0, 0 ], # values X,Y,Z values in voxels
        chunk_size = [ 1024,1024,1 ], # rechunk of image X,Y,Z in voxels -- only used for downsampling task I think
        volume_size = volume_size, # X,Y,Z size in voxels
        )

    vol = CloudVolume(f'file://{layer_dir}', info=info)
    vol.provenance.description = "Test on spock for profiling precomputed creation"
    vol.provenance.owners = ['ahoag@princeton.edu'] # list of contact email addresses
    if commit:
        vol.commit_info() # generates info json file
        vol.commit_provenance() # generates provenance json file
        print("Created CloudVolume info file: ",vol.info_cloudpath)
    if atlas_type:
        assert atlas_type in ['Princeton','Allen','Brodylab']
        info_dict = vol.info
        info_dict['atlas_type'] = atlas_type
        print(info_dict)
        info_filename = '/'.join(vol.info_cloudpath.split('/')[2:]) 
        with open(info_filename,'w') as outfile:
            json.dump(info_dict,outfile,sort_keys=True,indent=2)
        print("ammended info file to include 'atlas_type' key")
    return vol

def make_downsample_tasks(vol,mip_start=0,num_mips=3):
    """ 
    ---PURPOSE---
    Make downsamples of the precomputed data
    ---INPUT---
    vol             The cloudvolume.Cloudvolume() object
    mip_start       The mip level to start at with the downsamples
    num_mips        The number of mip levels to create, starting from mip_start
    """
    cloudpath = vol.cloudpath
    tasks = tc.create_downsampling_tasks(
        cloudpath, 
        mip=mip_start, # Start downsampling from this mip level (writes to next level up)
        fill_missing=False, # Ignore missing chunks and fill them with black
        axis='z', 
        num_mips=num_mips, # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
        chunk_size=[ 128, 128, 64 ], # manually set chunk size of next scales, overrides preserve_chunk_size
        preserve_chunk_size=True, # use existing chunk size, don't halve to get more downsamples
      )
    return tasks

def process_slice(z):
    if os.path.exists(os.path.join(progress_dir, str(z))):
        print(f"Slice {z} already processed, skipping ")
        return
    if z > (len(sorted_zplanes) - 1):
        print("Index {z} is larger than (number of slices - 1), skipping")
        return
    print('Processing slice z=',z)
    img_name = sorted_zplanes[z]
    image = Image.open(img_name)
    width, height = image.size 
    array = np.array(image, dtype=np.uint16, order='F')
    array = array.reshape((1, height, width)).T
    vol[:,:, z] = array
    image.close()
    touch(os.path.join(progress_dir, str(z)))
    print("success")
    return

if __name__ == "__main__":
    """ First command line arguments """
    # 1.1x data
    step = sys.argv[1]
    viz_dir = '/jukebox/LightSheetData/lightserv_testing/neuroglancer/brodyatlas'
    data_path = '/jukebox/LightSheetData/brodyatlas/testing/w122_43p_low_thres/full_sizedatafld/w122_tetrode_1_1x_488_555_008na_1hfds_z10um_100msec_40povlp_ch00'
    assert os.path.exists(data_path)
    layer_name = 'rawdata_w122_43p_low_thresh'
    layer_dir = os.path.join(viz_dir,layer_name)
    """ Make progress dir """
    progress_dir = mkdir(viz_dir + f'/progress_{layer_name}') # unlike os.mkdir doesn't crash on prexisting 
    all_zplanes = glob.glob(data_path + '/*tif')
    sorted_zplanes = sorted(all_zplanes)

    first_zplane = Image.open(sorted_zplanes[0])
    x_dim,y_dim = first_zplane.size
    z_dim = len(sorted_zplanes)
    x_scale_nm, y_scale_nm,z_scale_nm = 5909, 5909, 10000

    """ Handle the different steps """
    if step == 'step0':
        print("step 0")
        volume_size = (x_dim,y_dim,z_dim)
        resolution = (x_scale_nm,y_scale_nm,z_scale_nm)
        vol = make_info_file(volume_size=volume_size,
            layer_dir=layer_dir,resolution=resolution)
    elif step == 'step1':
        print("step 1")
        # Find the individual z planes in the full_sizedatafld - these are the blended images at the raw resolution
        vol = CloudVolume(f'file://{layer_dir}')
        done_files = set([ int(z) for z in os.listdir(progress_dir) ])
        all_files = set(range(vol.bounds.minpt.z, vol.bounds.maxpt.z + 1))

        to_upload = [ int(z) for z in list(all_files.difference(done_files)) ]
        to_upload.sort()
        print(f"Have {len(to_upload)} planes to upload")
        with ProcessPoolExecutor(max_workers=4) as executor:
            for job in executor.map(process_slice,to_upload):
                try:
                    print(job)
                except Exception as exc:
                    print(f'generated an exception: {exc}')
    elif step == 'step2': # downsampling
        print("step 2")
        vol = CloudVolume(f'file://{layer_dir}')
        tasks = make_downsample_tasks(vol,mip_start=0,num_mips=4)
        with LocalTaskQueue(parallel=4) as tq:
            tq.insert_all(tasks)


