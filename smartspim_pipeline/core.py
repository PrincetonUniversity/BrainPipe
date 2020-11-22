#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:50:20 2020

@author: wanglab
"""

import argparse, os
from argparse import RawDescriptionHelpFormatter
from spim_functions import csv_to_dict

def main(**args):
    
    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)    
    
    if params["stepid"] == 0:
        #######################################STITCH######################################################
        #save params
        save_params(params, params["input_path"])
        #stitch channels
        #pick job by array
        volin = params["volin"][params["jobid"]]
        dest = stitch(volin, params["xy"], params["z"], os.path.join(volin,"stitched"))
        print("\n\n***************************************\n"
              "stitched images!"
              "\n***************************************\n\n")
        
    # elif args["stepid"] == 1:
    #   #######################################CORRECTION after GENERATE FLAT###################################################
        
    
    elif args["stepid"] == 2:
        #####################################DOWNSIZE#################################
        
        #downsize one channel at a time
        channel1_downsized = downsize(params["channel1"],params["resizefactor"],params["cores"],params["atl"])
        params["channel1_downsized"]  = channel1_downsized
        if params["channel2"]:
            channel2_downsized = downsize(params["channel2"],params["resizefactor"],params["cores"],params["atl"])
            params["channel2_downsized"]  = channel2_downsized
        if params["channel3"]:
            channel3_downsized = downsize(params["channel3"],params["resizefactor"],params["cores"],params["atl"])
            params["channel3_downsized"]  = channel3_downsized
        if params["channel4]:
            channel4_downsized = downsize(params["channel4"],params["resizefactor"],params["cores"],params["atl"])
            params["channel4_downsized"]  = channel4_downsized
        
        #save out paths to the new downsized volumes
        save_params(params, params["input_path"])
        print("\n\n***************************************\n"
              "downsized images!"
              "\n***************************************\n\n")
        
    elif args["stepid"] == 3:
        ##############################################REGISTER#####################################################   
        
        #reload saved params
        params = csv_to_dict(os.path.join(params["input_path"], "spim_param_dict.csv"))
        
        #register based on jobid
        


def fill_params(**args):

    # print(args)
    params = {}

    #slurm params
    params["stepid"]            = args["stepid"]
    # params["jobid"]         = args.jobid 
    params["cores"]             = args["cores"]
    
    #experiment params
    params["input_path"]        = args["input_path"]
    #set channels
    params["channel1"]          = os.path.join(args["input_path"], args["channel1"])
    params["volin"]             = [params["channel1"]]
    if args.channel2 is not None:
        params["channel2"]      = os.path.join(args["input_path"], args["channel2"])
        params["volin"].append(params["channel2"])
    if args.channel3 is not None:
        params["channel3"]      = os.path.join(args["input_path"], args["channel3"])
        params["volin"].append(params["channel3"])
    if args.channel4 is not None:
        params["channel4"]      = os.path.join(args["input_path"], args["channel4"])
        params["volin"].append(params["channel4"])
    
    #pixel resolution
    params["xysize"]            = args["xysize"]
    params["zstep"]             = args["zstep"]
    params["resizefactor"]      = args["resizefactor"]
    #registration
    params["registration"]      = args["registration"]
    if args["registration"]:
        params["orientation"]   = args["orientation"]
        params["parameterfld"]  = args["parameterfld"]
        params["atlas"]         = args["atlas"]
        
    return params

def save_params(params, dst):
    """ 
    save params 
    can discard later if need be
    """
    (pd.DataFrame.from_dict(data=params, orient="index").to_csv(os.path.join(dst, "spim_param_dict.csv"),
                            header = False))
    print("\nparameters saved in: {}".format(os.path.join(dst, "spim_param_dict.csv")))
    
    return
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BrainPipe for smartSPIM\n\n",
                                     formatter_class=RawDescriptionHelpFormatter,
                                     epilog="2020 by Zahra Dhanerawala, Wang Lab\n"
                                            "Princeton Neuronscience Institute\n")
    parser.add_argument("--stepid", "-s", help="Step of pipeline to run (out of 4)", type=int, required=True)
    parser.add_argument("--input_path", "-i", help="Path to microscope image folder (NOT channel subfolder)", type=str, required=True)
    parser.add_argument("--channel1", "-ch1", help="First channel/wavelength imaged", type=str, default="Ex_488_Em_0")
    parser.add_argument("--channel2", "-ch2", help="Second channel/wavelength imaged", type=str, default=None) #"Ex_647_Em_2"
    parser.add_argument("--channel3", "-ch3", help="Third channel/wavelength imaged", type=str, default=None)
    parser.add_argument("--channel4", "-ch4", help="Fourth channel/wavelength imaged", type=str, default=None)
    parser.add_argument("--xysize", "-xy", help=" XY pixel resolution (micron)", type=float, default=1.81)
    parser.add_argument("--zstep", "-z", help=" Z step (micron)", type=int, default=2)
    parser.add_argument("--cores", "-c", help="# of cores used", type=int, default=12)
    parser.add_argument("--resizefactor", "-rf", help="Resize factor for downsized volumes", type=int, default=5)
    parser.add_argument("--registration", "-reg", help="If registration needs to be done", type=bool, default=True)
    parser.add_argument("--parameterfld", "-p", help="Folder containing elastix-format parameters for registration", type=str, 
                        default="/jukebox/wang/zahra/python/BrainPipe/parameterfolder")
    parser.add_argument("--atlas", "-a", help="Atlas used for registration", type=str, 
                        default="/jukebox/LightSheetTransfer/atlas/sagittal_atlas_20um_iso.tif")
    parser.add_argument("--orientation", "-or", help="Final orientation used for registration", type=str, default="sagittal")
    args = parser.parse_args()
    
    main(**vars(args))    
    
    
        
    