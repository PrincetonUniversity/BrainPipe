# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:07:30 2016

@author: tpisano
"""
import os, socket

def directorydeterminer():
    """Function to allow for different paths to the same server. This allows for working locally and on cluster.

    EDIT THIS to fit your paths.
    """
    if socket.gethostname() == "spock-login.pni.princeton.edu":
        systemdirectory= "/jukebox/"
    elif socket.gethostname() == "wang-38vfpd2":
        systemdirectory= "/home/wanglab/"
    elif socket.gethostname() == "pni-3cnxk02":
        systemdirectory= "/home/tpisano/"
    elif socket.gethostname() == "wanglab-cr8rc42-ubuntu":
        systemdirectory=  "/home/wanglab/"
    elif socket.gethostname() == "PNI-1867VWTQ2":#zmd added
        systemdirectory = "/jukebox/"
    elif os.getcwd()[:6] == "/Users":
        systemdirectory= "/Volumes/"
    elif os.getcwd()[:8] == "/Volumes":
        systemdirectory= "/Volumes/"
    elif os.getcwd()[:8] == "/jukebox":
        systemdirectory= "/jukebox/"
    elif os.getcwd()[:4] == "/mnt":
        systemdirectory= "/jukebox/"
    else:
        systemdirectory= "/jukebox/"
    return systemdirectory
