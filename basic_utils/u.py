#!/usr/bin/env python

#Basic python utilities


import numpy as np
import h5py
import tifffile
import os


def read_bin(fname):
	d = np.fromfile(fname, dtype='double')
	shape = np.fromfile(fname + '.size', dtype='uint32')[::-1]
	return d.reshape(shape)

def write_bin(data, fname):
	shape = np.array(data.shape).astype('uint32')
	data.tofile(fname)
	shape.tofile(fname + '.size')

def read_tif(fname):
	return tifffile.imread(fname)

def write_tif(data, fname):
	tifffile.imsave(fname, data)

def read_h5(fname):
	f = h5py.File(fname)
	d = f['/main'].value
	f.close()
	return d

def write_h5(data, fname):

	if os.path.exists(fname):
		os.remove(fname)

	f = h5py.File(fname)
	f.create_dataset('/main',data=data)
	f.close()

def h5write(data, fname, path="/main"):
        "Version of write_h5 for writing multiple dsets to a single file"

        f = h5py.File(fname)
        f.create_dataset(path, data=data)
        f.close()

def write_chunked_h5(data, fname):

	if os.path.exists(fname):
		os.remove(fname)

	f = h5py.File(fname)
	f.create_dataset('/main',data=data, chunks=(128,128,128), compression='gzip',compression_opts=4)
	f.close()

def read_file(fname):
	if '.h5' in fname:
		return read_h5(fname)
	elif '.tif' in fname:
		return read_tif(fname)
	elif os.path.isfile( fname + '.size'):
		return read_bin(fname)
	else:
		print("Unsupported file? (no .tif or .h5)")

def read_files(*fnames):
	return [read_file(f) for f in fnames]

def write_file(data, fname):
	if '.h5' in fname:
		write_h5(data, fname)
	elif '.tif' in fname:
		write_tif(data, fname)
	else:
		print("Unsupported file? (no .tif or .h5)")
