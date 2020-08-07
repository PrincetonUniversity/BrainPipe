import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
import tifffile

from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

home_dir = '/home/emilydennis/Desktop/GitHub/ng'
atlas_file = '/home/ahoag/progs/brodylab/atlas/result.1.tif'


def make_info_file(commit=True):
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='image',  # 'image' or 'segmentation'
        data_type='uint16',  # 32 not necessary for Princeton atlas, but was for Allen atlas
        encoding='raw',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution=[25000, 25000, 25000],  # X,Y,Z values in nanometers, 40 microns in each dim
        voxel_offset=[0, 0, 0],  # values X,Y,Z values in voxels
        chunk_size=[1024, 1024, 1],  # rechunk of image X,Y,Z in voxels
        volume_size=[390, 1150, 618],  # X,Y,Z size in voxels
    )

    # If you're using amazon or the local file system, you can replace 'gs' with 's3' or 'file'
    vol = CloudVolume('file:///home/ahoag/ngdemo/demo_bucket/atlas/brodyatlas_2019_w128', info=info)
    vol.provenance.description = "Brodylab test atlas image volume"
    vol.provenance.owners = ['ahoag@princeton.edu']  # list of contact email addresses
    if commit:
        vol.commit_info()  # generates gs://bucket/dataset/layer/info json file
        vol.commit_provenance()  # generates gs://bucket/dataset/layer/provenance json file
        print("Created CloudVolume info file: ", vol.info_cloudpath)
    return vol


def process_slice(z):
    print('Processing slice z=', z)
    array = image[z].reshape((1, y_dim, x_dim)).T
    vol[:, :, z] = array
    touch(os.path.join(progress_dir, str(z)))
    return "success"


if __name__ == '__main__':
    """ Fill the CloudVolume() instance with data from the tif slices """
    vol = make_info_file()
    """ Now load the tifffile in its entirety """
    image = np.array(tifffile.imread(atlas_file), dtype=np.uint16,
                     order='F')  # F stands for fortran order
    z_dim, y_dim, x_dim = image.shape
    # unlike os.mkdir doesn't crash on prexisting
    progress_dir = mkdir(home_dir + '/progress_dirs/progress_brodyatlas_2019_w128')

    done_files = set([int(z) for z in os.listdir(progress_dir)])
    all_files = set(range(vol.bounds.minpt.z, vol.bounds.maxpt.z))

    to_upload = [int(z) for z in list(all_files.difference(done_files))]
    to_upload.sort()
    print("Remaining slices to upload are:", to_upload)

    with ProcessPoolExecutor(max_workers=8) as executor:
        for job in executor.map(process_slice, to_upload):
            try:
                print(job)
            except Exception as exc:
                print(f'generated an exception: {exc}')

    vol.cache.flush()
