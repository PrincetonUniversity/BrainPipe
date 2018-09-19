#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.external.tifffile import TiffFile
import h5py
from os.path import basename, splitext, isfile
import argparse

def convert_tiff(tiff_filename):
    with TiffFile(tiff_filename) as tif:
        if not isfile(tiff_filename):
            error_string = "{} does not exist".format(basename(tiff_filename))
            raise EnvironmentError(error_string)
        
        h5_filename = splitext(tiff_filename)[0] + ".h5"

        with h5py.File(h5_filename) as h5_file:
            h5_file.create_dataset("/main", data=tif.asarray())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Converts a TIFF stack to a" +
                                     " HDF5 dataset for training")

    parser.add_argument('FILE')

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        convert_tiff(args.FILE)

    except EnvironmentError:
        print("error: TIFF file does not exist")

if __name__ == '__main__':
    main()
