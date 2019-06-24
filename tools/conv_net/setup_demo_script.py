import numpy as np, os, sys
import tifffile
import h5py

if __name__ == '__main__':
    
    #get location
    location_to_save = sys.argv[1]
    print(location_to_save)
    
    #make dirs
    if not os.path.exists(location_to_save): os.mkdir(location_to_save)
    pth = os.path.join(location_to_save, 'input_patches')
    if not os.path.exists(pth): os.mkdir(pth)
    
	#generate dummy array:
    arr = np.random.rand(30,200,200)*100
    tifffile.imsave(os.path.join(pth, 'demo.tif'), arr.astype('float32'), compress=1)
    
    #generate train dataset:
    with h5py.File(os.path.join(location_to_save, 'train_img.h5'), 'w') as f:
        train = f.create_dataset('/main', data=arr.astype('float32'))
    with h5py.File(os.path.join(location_to_save, 'train_lbl.h5'), 'w') as f:
        train = f.create_dataset('/main', data=arr.astype('float32'))
    
    #generate val dataset
    with h5py.File(os.path.join(location_to_save, 'val_img.h5'), 'w') as g:
        val = g.create_dataset('/main', data=arr.astype('float32'))
    with h5py.File(os.path.join(location_to_save, 'val_lbl.h5'), 'w') as g:
        val = g.create_dataset('/main', data=arr.astype('float32'))


