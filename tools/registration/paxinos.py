"""
Created on Mon Mar  6 10:45:19 2017

@author: ejdennis based on ahoag

Goal is to transform Waxholm coords (and eventually PRA coords) into Paxinos

To do:
TEST

"""

import neuroglancer
import cloudvolume
import json
import csv
import graphviz
import os
import sys


if __name__ == '__main__':
    neuroglancer.set_static_content_source(
        url='https://nglancer.pni.princeton.edu')
    # neuroglancer.set_static_content_source(url='http://localhost:8080')
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers['atlas'] = neuroglancer.SegmentationLayer(
            source='precomputed://http://localhost:1338')
    print(viewer)

    num_actions = 0
    viewer.actions.add('my-action', my_action)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer['keyp'] = 'my-action'
    #     s.status_messages['hello'] =
    # 'Welcome to the parent merge first example. Press p to use.'

    with viewer.txn() as s:
        print(s)

    with viewer.txn() as s:
        s.layout = "yz"


def MRITopaxinos_AP(MRIAP):
    return(-((MRIAP-214)*25-20)/1000)


def MRITopaxinos_DV(MRIDV):
    return(25/1000.*MRIDV)


def MRITopaxinos_ML(MRIZ):
    return(25/1000.*(456/2.-MRIZ))


def my_action(s):
    global num_actions
    num_actions += 1
    with viewer.config_state.txn() as st:
        #         print('  Mouse position: %s' % (s.mouse_voxel_coordinates,))
        mouse_position_tuple = s.mouse_voxel_coordinates
        x, y, z = mouse_position_tuple
        ML = allenTopaxinos_ML(z)
        AP = allenTopaxinos_AP(y)
        DV = allenTopaxinos_DV(x)
        print("X,Y,Z:")
        print(x, y, z)
        print("ML, DV, AP:")
        print(ML, DV, AP)
#         print(len(mouse_position_tuple))
#         print('  Layer selected values: %s' % (s.selected_values,))
