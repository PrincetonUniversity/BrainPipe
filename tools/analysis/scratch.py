
from rat_atlas.step3_make_median import generate_median_image, load_memmap_arr


import os
import numpy as np
import tifffile as tif
import sys
from scipy.ndimage import zoom
sys.path.append("/home/emilyjanedennis/Desktop/GitHub/rat_BrainPipe/")
from tools.registration.register import elastix_command_line_call
src = "/home/emilyjanedennis/Desktop/vols"
param_fld = "/home/emilyjanedennis/Desktop/affine/"

mvtiffs = ["f110","c514","t107","e106"]
fxtiffs = ["k320","f003","a235","f002"]

for pairnum in np.arange(0,len(mvtiffs)+1):
	mvtiff = mvtiffs[pairnum]
	fxtiff = fxtiffs[pairnum]
    output_folder = os.path.join(src,"out")
    if not os.path.exists(output_fld): os.mkdir(output_fld)

    final_output_path = os.path.join(src,"new_medians/{}".format(pairnum))
    if not os.path.exists(output_fld): os.mkdir(final_output_path)

    parameters=1
    memmappth = os.path.join(src,"out/{}_to_{}_mm.npy".format(mvtiff,fxtiff))

    generate_median_image(output_fld, parameters, memmappth, final_output_path, verbose = True)
