#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:11:59 2018

@author: wanglab
"""

import subprocess as sp


def transfer(src, dest, label, pull = True, other_endpoint = False):
    '''Tranfers files from tigress to local/jukebox locations.
    Inputs:
        src = tigress source path
        dest = destination path
        label = name of transfer
        pull = True; specify whether you are pushing to tigress or pulling from tigress *not functional yet
        other_endpoint = False; assumes transfer to wanglab A84D computer unless jukebox/other endpoint keys specified
    '''    
    #for globus transfer, endpoint keys: 
    #tigress = 'a9df83d2-42f0-11e6-80cf-22000b1701d1' #transferring from
    #wanglab = 'f7949748-c728-11e8-8c57-0a1d4c5c824a' #transferring to
        
    tigress_pth = 'a9df83d2-42f0-11e6-80cf-22000b1701d1:'+src
    if not other_endpoint:
        local_pth = 'f7949748-c728-11e8-8c57-0a1d4c5c824a:'+dest
    else:
        local_pth = other_endpoint+':'+dest
    
    sp.call(['globus', 'transfer', tigress_pth, local_pth, '--recursive', '--label', label]) #run command line call
    
if __name__ == '__main__':
    
    src = '/tigress/zmd/wang/zahra/3dunet_cnn/experiments/20181001_zd_train/forward'
    dest = 'Documents/python/data/20181001_zd_train/52000chkpnt' #wanglab endpoint assumes default directory of /home/wanglab
    label = 'model_20181001'
    transfer(src, dest, label, other_endpoint = False)
    