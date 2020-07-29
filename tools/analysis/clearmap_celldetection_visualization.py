#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Feb 20 12:43:10 2019



@author: wanglab

"""



from skimage.external import tifffile as tif

from skimage.morphology import ball

from scipy.ndimage.interpolation import zoom

import matplotlib.pyplot as plt, numpy as np, cv2, os, pandas as pd

import matplotlib.colors

from matplotlib.backends.backend_pdf import PdfPages



#%matplotlib inline



##########################################################################RUNS IN PYTHON 3###############################################################

class Sagittal():

    """ 

    accepts sagittal volumes and can make z projected sections and overlay volumes for quality control and visualisation

    NOTE: cells destination must be .npy (z,y,x) for now

    """

    

    def __init__(self, src, dst, cells):

        self.src = src

        self.saggital = tif.imread(src)

        self.coronal = np.transpose(self.saggital, [1, 0, 2])

        self.horizontal = np.transpose(self.saggital, [2, 1, 0])

        if cells[-4:] == ".npy": self.cells = np.load(cells)

        elif cells[-4:] == ".csv": self.cells = pd.read_csv(cells)

        self.dst = dst

        print(os.path.dirname(os.path.dirname(src)))

        

    def makeSagittalSections(self, save = False):

        #make subplots

        fig, ax = plt.subplots(1, 6)

        #set chunk size

        chunk = 30

        #for loop

        slice = 350

        for n in range(6):

            a = np.max(self.saggital[slice-chunk:slice, :, :]*3, axis = 0)

            ax[n].imshow(a, "gist_yarg")

            ax[n].set_xticks([])

            ax[n].set_yticks([])

            ax[n].set_title("z = {}".format(slice), color="m", fontsize = 10)

            slice += chunk

           

        fig.subplots_adjust(wspace=0, hspace=0)

        if save: plt.savefig(os.path.join(self.dst, "{}_sagittal.pdf".format(os.path.basename(os.path.dirname(os.path.dirname(self.src))))), dpi = 300)

        else: plt.show()

        

    

    def makeCoronalSections(self, save = False): 

        """

        making coronal volume from sagittal volumes

        inputs:

            save = T/F

        """



        print(self.coronal.shape)

        

        fig, ax = plt.subplots(1, 6)

        #set chunk size 

        chunk = 30

        slice = 200

        for n in range(6):

            a = np.max(self.coronal[slice-chunk:slice, :, :]*3, axis = 0)

            ax[n].imshow(np.rot90(a, axes = (1,0)), "gist_yarg", fontsize = 10)

            ax[n].set_xticks([])

            ax[n].set_yticks([])

            ax[n].set_title("z = {}".format(slice), color="m")

            slice += chunk

           

        fig.subplots_adjust(wspace=0, hspace=0)

        if save: plt.savefig(os.path.join(self.dst, "{}_coronal.pdf".format(os.path.basename(os.path.dirname(os.path.dirname(self.src))))), dpi = 300)

        else: plt.show()

        

    def makeHorizontalSections(self, save = False):

        """

        making horizontal volume from sagittal volumes

        inputs:

            save = T/F

        """



        print(self.horizontal.shape)

        

        fig, ax = plt.subplots(2, 3)

        #set chunk size 

        chunk = 30

        slice = 150

        for m in range(2):

            for n in range(3):

                a = np.max(self.horizontal[slice-chunk:slice, :, :]*10, axis = 0)

                ax[m, n].imshow(a, "gist_yarg")

                ax[m, n].set_xticks([])

                ax[m, n].set_yticks([])

                ax[m, n].set_title("z = {}".format(slice), color="m", fontsize = 10)

                slice += chunk

           

        fig.subplots_adjust(wspace=0)

        if save: plt.savefig(os.path.join(self.dst, "{}_horizontal.pdf".format(os.path.basename(os.path.dirname(os.path.dirname(self.src))))), dpi = 300)

        else: plt.show()

        

        

    def makeClearMapCellOverlayHorizontalSections(self, volume = False, save = True):

        """

        making horizontal volumes with cells detected from sagittal volumes

        inputs:

            save = T/F

        """



        print("\nhorizontal resampled image shape: {}\n".format(self.horizontal.shape))

        print("number of cells detected: {}\n".format(self.cells.shape[0]))

        

        #make overlay

        alpha = 0.6 #determines transparency, don't need to alter

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "lime"]) #lime color makes cells pop

#        cmap.set_under("white", alpha = 0)

        

        #first, make a map of cells

        zyx = np.asarray([(int(xx[2]), int(xx[1]), int(xx[0])) for xx in self.cells]) #cells are counted in horizontal volumes

            

        #get full size dimensions - have to do this without importing pickle

        fullszflds = [os.path.join(os.path.join(os.path.dirname(os.path.dirname(self.src)), "full_sizedatafld"), xx) 

                        for xx in os.listdir(os.path.join(os.path.dirname(os.path.dirname(self.src)), "full_sizedatafld"))]

        y,x = tif.imread(os.path.join(fullszflds[0], os.listdir(fullszflds[0])[0])).shape

        

        z = len(os.listdir(fullszflds[0])) 

        self.fullszdims = (z,y,x)

        cell_map = np.zeros(self.fullszdims, dtype = "bool") 

        for z,y,x in zyx:

            try:

                cell_map[z-1:z+1,y,x] = True

            except Exception as e:

                print(e)

                

        #apply x y dilation

        r = 2

        selem = ball(r)[int(r/2)]

        cell_map = cell_map.astype("uint8")

        cell_map = np.asarray([cv2.dilate(cell_map[i], selem, iterations = 1) for i in range(cell_map.shape[0])])

        

        #resampling cells

        resizef = (1/1.6, 1/9.81, 1/9.81) #in z,y,x, depends on resampling, maybe better not to be hardcoded; always resampled this way in ClearMap

        print("resizing by factors (z,y,x): {}\n this can take some time...\n".format(resizef))

        resz_cell_map = zoom(cell_map, resizef, order = 1) #right now only linear interpolation

        print("horizontal resampled cell map shape: {}\n".format(resz_cell_map.shape))

         

        #if selected to save volumteric RBG image:

        if volume: 

            if self.horizontal.shape[2] > resz_cell_map.shape[2]: #messy way to do this but have to to adjust for scipy zoom

                self.horizontal = self.horizontal[:, :, :(self.horizontal.shape[2]-1)]

                print("horizontal resampled image shape: {}\n".format(self.horizontal.shape))

            elif self.horizontal.shape[2] < resz_cell_map.shape[2]:

                resz_cell_map = resz_cell_map[:, :, :(resz_cell_map.shape[2]-1)]

                print("horizontal resampled cell map shape: {}\n".format(resz_cell_map.shape))

            if self.horizontal.shape[0] < resz_cell_map.shape[0]:

                resz_cell_map = resz_cell_map[:(resz_cell_map.shape[0]-1), :, :]

                print("horizontal resampled cell map shape: {}\n".format(resz_cell_map.shape))

            elif self.horizontal.shape[0] > resz_cell_map.shape[0]:

                self.horizontal = self.horizontal[:(self.horizontal.shape[0]-1), :, :]

                print("horizontal resampled image shape: {}\n".format(self.horizontal.shape))

            

            merged = np.stack([self.horizontal, resz_cell_map, np.zeros_like(self.horizontal)], -1)

            tif.imsave(os.path.join(self.dst, "{}_points_merged.tif".format(os.path.basename(os.path.dirname(os.path.dirname(self.src))))), merged)

            

        #compiles into multiple pdfs

        pdf_pages = PdfPages(os.path.join(self.dst, "{}_cell_overlay_horizontal.pdf".format(os.path.basename(os.path.dirname(os.path.dirname(self.src)))))) 

        

        #set chunk size 

        chunk = 30

        slice = 200

        for n in range(6):

            #open figure

            plt.figure(figsize=(8.27, 11.69))

            a = np.max(self.horizontal[slice-chunk:slice, :, :]*30, axis = 0) #the * factor is something you have to test and see what looks good, coudl be a variable

            b = np.max(resz_cell_map[slice-chunk:slice, :, :]*3,axis = 0)

            plt.imshow(a, "gist_yarg")

            plt.imshow(b, cmap, alpha = alpha)

            plt.axis("off")

            plt.title("z = {}".format(slice), color="m", fontsize = 10)

            

            #done with the page

            if save: pdf_pages.savefig(dpi = 300, bbox_inches = 'tight')

            plt.close()

            slice += chunk

           

        if save: pdf_pages.close()

        

        print("done!\n*********************************************************************")

    

    

#%%

if __name__ == "__main__":

    #grabbing sagittal volume

    dst = "/jukebox/LightSheetData/rat-brody/processed/201910_tracing/qc"

    if not os.path.exists(dst): os.mkdir(dst)



    pth = "/jukebox/LightSheetData/rat-brody/processed/201910_tracing/clearmap"

    # flds = os.listdir(pth)

    

    # for fld in flds:

    fld = "z265"

    src = os.path.join(pth, fld+"/clearmap_cluster_output/cfos_resampled.tif")

    cells = os.path.join(pth, fld+"/clearmap_cluster_output/cells.npy")

    if os.path.exists(cells):

        sagittal = Sagittal(src, dst, cells)

        sagittal.makeClearMapCellOverlayHorizontalSections(volume = False, save = True)
