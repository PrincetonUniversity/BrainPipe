# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:07:30 2016

@author: tpisano


Function to allow for different paths to the same server. For working on different servers/computers.

Alternative would be to softlink paths of different computers to have the same paths as cluster. I.e. /jukebox/ for both computer and SPOCK

"""
import os, socket, collections

def directorydeterminer():
    """Function to allow for different paths to the same server. This allows for working locally and on cluster.

    EDIT THIS to fit your paths.
    """
    if socket.gethostname() == "spock-login.pni.princeton.edu":
        systemdirectory= "/jukebox/"
    if socket.gethostname() == "PNI-1867vwtq2":
        systemdirectory = "/jukebox/"
    else:
        systemdirectory = "/jukebox/"

    return systemdirectory


def pth_update(item):    
    """simple way to update dictionary, list, or str for local path, should be recursive for dicts
    """
    for prefix in ["/jukebox/", "/mnt/bucket/labs/", "/home/wanglab/", "/home/tpisano/", "/scratch/gpfs/tpisano/", "/tigress/tpisano/"]:
        if type(item) == dict:
            for keys, values in item.items():
                if type(values) == str:
                    if prefix in values:
                        item[keys] = os.path.join(directorydeterminer(), values[values.rfind(prefix)+len(prefix):])
                elif type(values) == dict:
                    item[keys]=dict_recursion(values, prefix)
                elif type(values) == list:
                    item[keys] = update_list(values, prefix)
        elif type(item) == list:
            nlst = []
            for i in item:
                if type(i) == str:
                    if prefix in i:
                        nlst.append(os.path.join(directorydeterminer(), i[i.rfind(prefix)+len(prefix):]))
                    else:
                        nlst.append(i)
                else:
                    nlst.append(i)
            item = nlst
        elif type(item) == str and prefix in item:
            item = os.path.join(directorydeterminer(), item[item.rfind(prefix)+len(prefix):])        
    return item


def dict_recursion(d, prefix):
    nd = {}
    for k, v in d.items():
        if prefix in k: k = os.path.join(directorydeterminer(), k[k.rfind(prefix)+len(prefix):])
        if isinstance(v, dict):
            dict_recursion(v)
        else:
            if type(v) == str:      
                if prefix in v: v = os.path.join(directorydeterminer(), v[v.rfind(prefix)+len(prefix):])
        nd[k] = v
    return nd

def update_list(lst, prefix):
    nlst = []
    for v in lst:
        if isinstance(v, collections.Iterable) and prefix in v:
            nlst.append(os.path.join(directorydeterminer(), v[v.rfind(prefix)+len(prefix):]))
        else:
            nlst.append(v)
    return nlst





