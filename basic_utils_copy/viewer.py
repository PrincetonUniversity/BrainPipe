#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = """
Created on Wed Jan 28 14:28:47 2015

@author: jingpeng, nick
"""

import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors

class Viewer(object):

    def __init__(self, vols, cmap='gray'):

        #zero-padded copies of the volumes
        self.vols = self.__pad(vols)

        #Number of slices to display
        self.Nz = min([elem.shape[0]-1 for elem in vols])
        #Current z index
        self.z = 0

        #Colormap argument to be passed to imshow (set under 'c' keypress)
        self.cmap = [cmap for elem in vols]
        self.norm = [[] for elem in vols]
        #Whether or not the plot at an index is a color plot
        self.colorplot = [False for elem in vols]

        #Defining a current index of the vols to be 'selected' for
        # modification by (for example) adding color
        self.selected = 0

        self.__show()


    def swap_volumes(self, new_vols):

        new_vols = self.__pad(new_vols)

        self.Nz = min([elem.shape[0]-1 for elem in new_vols])
        self.z = min(self.z, self.Nz)

        self.vols = new_vols

        self.__refresh()


    def __show(self):
        self.fig, self.axs = plt.subplots(1,len(self.vols), sharex=True, sharey=True)
        self.fig.canvas.mpl_connect('key_press_event', self.__press)

        if len(self.vols) > 1:
          for i in range(len(self.vols)):
              axis = self.axs[i]
              self.__show_slice(axis, self.vols[i][self.z,:,:],
                                cmap=self.cmap[i], norm=self.norm[i])
              axis.set_xlabel( ' volume {0}: slice {1}'.format(i+1,self.z) )
        else:
          axis = self.axs
          self.__show_slice(self.axs, self.vols[0][self.z,:,:],
                            cmap=self.cmap[0], norm=self.norm[0])
          axis.set_xlabel( ' volume {0}: slice {1}'.format(1,self.z) )

        if __name__ != "__main__":
          plt.ion()
        plt.show()


    def __show_slice(self, axis, imslice, cmap, norm):

        #normed_slice = self.__norm(sl)
        if cmap == "gray":
          axis.imshow(imslice, interpolation="nearest", cmap=cmap)
        else:
          axis.imshow(imslice, interpolation="nearest", cmap=cmap, norm=norm)


    def __refresh(self):
        '''Basic refresh function'''

        if len(self.vols) > 1:
          for i in range(len(self.axs)):
              axis = self.axs[i]
              axis.images.pop()
              self.__show_slice(axis, self.vols[i][self.z,:,:],
                                cmap=self.cmap[i], norm=self.norm[i])
              axis.set_xlabel( ' volume {}: slice {}'.format(1,self.z) )

        else:
          axis = self.axs
          axis.images.pop()
          self.__show_slice(axis, self.vols[0][self.z,:,:],
                            cmap=self.cmap[0], norm=self.norm[0])
          axis.set_xlabel( ' volume {}: slice {}'.format(1,self.z) )

        self.fig.canvas.draw()


    def __pad(self, vols):
        '''Zero-padding a list of input volumes to match by non-z dimensions'''
        shapes = np.array(
            [elem.shape for elem in vols]
            )

        max_shape = np.max(shapes,0)

        pad_vols = [np.zeros((elem.shape[0], max_shape[1], max_shape[2]))
                    for elem in vols]

        dim_diffs = [(max_shape - elem.shape) // 2
                    for elem in vols]

        for i in range(len(pad_vols)):

            if all(dim_diffs[i][1:] != 0):
                pad_vols[i][
                    :,
                    dim_diffs[i][1]:-(dim_diffs[i][1]),
                    dim_diffs[i][2]:-(dim_diffs[i][2])
                    ] = vols[i]
            else:
                pad_vols[i] = vols[i]

        return pad_vols


    def __norm(self, imslice):
        #subtract the nonzero minimum from each slice
        nonzero_indices = np.nonzero(imslice)
        if len(nonzero_indices) > 0 and np.max(imslice) > 1:
            nonzero_min = np.min(imslice[np.nonzero(imslice)])

            res = np.copy(imslice)
            res[np.nonzero(res)] = res[np.nonzero(res)] - nonzero_min + 1
        else:
            res = imslice
        return res


    def __make_cmap(self, i):
        max_num_colors = 500

        #(0,0,0) = black
        plot_colors = np.vstack(((0,0,0), np.random.rand(max_num_colors,3)))
        cmap = colors.ListedColormap(plot_colors)

        norm = colors.Normalize(0,500)

        return cmap, norm


    def __press(self, event):
#       print 'press ' + event.key
        if 'down' == event.key and self.z<self.Nz:
            self.z+=1
        elif 'up' == event.key and self.z > 0: #>-self.Nz:
            self.z-=1
        elif 'c' == event.key:
            #Swap between color display and b&w
            self.colorplot[self.selected] = not self.colorplot[self.selected]

            if self.colorplot[self.selected]:
                new_cmap, new_norm = self.__make_cmap(self.selected)

                self.cmap[self.selected] = new_cmap
                self.norm[self.selected] = new_norm

            else:
                self.cmap[self.selected] = 'gray'

        elif 'j' == event.key:
            self.z += 10
            if self.z >= self.Nz:
                self.z = self.Nz - 1

        elif 'k' == event.key:
            self.z -= 10
            if self.z <= 0:
                self.z = 0

        elif 'v' == event.key:
            #Display the data values for the given data coordinate
            xcoord, ycoord = int(event.xdata), int(event.ydata)
            print((xcoord, ycoord))

            print([vol[self.z, ycoord, xcoord] for vol in self.vols])

        elif 'p' == event.key: #paint

            xcoord, ycoord = int(event.xdata), int(event.ydata)
            sl_shape = self.vols[0].shape[1:]
            print((xcoord, ycoord))

            xmin, xmax = max(xcoord-3,0), min(xcoord+3,sl_shape[1])
            ymin, ymax = max(ycoord-3,0), min(ycoord+3,sl_shape[0])

            self.vols[self.selected][self.z, ymin:ymax, xmin:xmax] = 999

        elif 'e' == event.key: #erase

            xcoord, ycoord = int(event.xdata), int(event.ydata)
            print((xcoord, ycoord))

            self.vols[self.selected][self.z, ycoord, xcoord] = 0

        elif event.key in ['1','2','3','4','5','6','7','8','9']:
            #Select a new axis
            index = int(event.key)

            if index - 1 < len(self.vols):
                self.selected = index-1

        self.__refresh()


def transpose(vol):
    rev_dims = tuple(reversed(range(len(vol.shape))))
    return vol.transpose(rev_dims)


def split_vols(vol):
    if len(vol.shape) == 3:
        return [vol]
    elif len(vol.shape) == 4:
        return [vol[i,...] for i in range(vol.shape[0])]
    else:
        raise Exception("vol with <3 or >4 dimensions")


if __name__ == "__main__":

    import u

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("vols", nargs="+")
    parser.add_argument("--transpose", action="store_true",
                        help="Transpose the volumes before viewing")

    args = parser.parse_args()
    
    raw_vols = [u.read_file(f) for f in args.vols]

    #preprocess multi-volumes
    vols = []
    for v in raw_vols:
        for split_vol in split_vols(v):
            vols.append(split_vol)

    if args.transpose:
        vols = [transpose(vol) for vol in vols]

    Viewer(vols)
