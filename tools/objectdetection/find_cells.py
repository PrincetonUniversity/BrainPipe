import cv2
_cv_ver = int(cv2.__version__[0])
import numpy as np
from skimage.external import tifffile
from skimage import feature
from skimage.exposure import equalize_adapthist as clahe
from skimage.morphology import square, dilation
import pylab as pl

def circularity(contours):
    """
    A Hu moment invariant as a shape circularity measure, Zunic et al, 2010
    """
    #moments = [cv2.moments(c.astype(float)) for c in contours]
    #circ = np.array([(m['m00']**2)/(2*np.pi*(m['mu20']+m['mu02'])) if m['mu20'] or m['mu02'] else 0 for m in moments])
    
    circ = [ (4*np.pi*cv2.contourArea(c))/(cv2.arcLength(c,True)**2) for c in contours]

    return np.asarray(circ)

def process_image(im):
    #c = clahe(im, ntiles_x=10, ntiles_y=10, clip_limit=0.5)
    #return ((2**16-1) * c).astype(np.uint16)    
    return im

def display_cells(im, cells, mode='contours'):
    # Display cells represented by *cells* on image *im*
    # if contours, cells are pts; if edges, cells are one edge img
    im = im.copy()

    if mode == 'contours':
        centers = np.asarray([c.mean(axis=0) for c in cells])
        cv2.drawContours(im, cells, -1, 3*(2**16-1,), thickness=1)
        for c in centers:
            pass
            #cv2.circle(im, tuple(c.astype(int)), 2, 3*(2**16-1,), thickness=3)
    elif mode == 'edges':
        im[cells.astype(bool)] = im.max()
    return im

def find_edges(im, edge_finding_param=0.85, sigma=4, dilation_kernel_size=5):
    #edge_finding_param : haven't come up with a better name for it, classic thing to help select edge-finding criteria
    med = np.median(im)
    th1,th2 = int(max(0, (1.0 - edge_finding_param) * med)), int(min(255, (1.0 + edge_finding_param) * med))
    edgeim = feature.canny(im, low_threshold=th1, high_threshold=th2, sigma=sigma)
    edgeim = edgeim.astype(np.uint8)

    edgeim = dilation(edgeim, square(dilation_kernel_size)) 

    return edgeim

def find_cells_intensity(edge_im, im_orig,  cell_area_thresh=[55,1750], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.9925, xyz_scale=(1,1,1)):
    """ Find cells

    Parameters
    -----------
    edge_im : 8- or 16-bit input image (single image, not stack)
    cell_area_thresh = [min, max] contour area, in microns
    cell_circularity_thresh = number from 0-1, where 1 means perfect circle
    xyz_scale: (x_microns_per_pixel, y..., z...); NOTE THIS IS USED TO CALCULATE THRESHOLDS, BUT OUTPUT IS IN PIXELS

    Returns
    -------
    centers : center of cells ##x,y NOTE THIS IS DIFFERENT THAN NP
    contours : contours of cells ##x,y

    """
    # find contours
    if _cv_ver == 2:
        contours,hierarchy = cv2.findContours(edge_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    elif _cv_ver == 3:
        cim,contours,hierarchy = cv2.findContours(edge_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # process contours
    contours = np.asarray([c.squeeze() for c in contours if cv2.contourArea(c)>0])
    carea = np.array(map(cv2.contourArea, contours))
    mean_intensities = np.zeros(len(contours))
    for i,c in enumerate(contours):
        minx,miny = c.min(axis=0)
        maxx,maxy = c.max(axis=0)
        mi = im_orig[miny:maxy, minx:maxx].max() ##mean is now max intensities**
        mean_intensities[i] = mi

    # convert params
    cell_area_thresh = np.asarray(cell_area_thresh)/np.asarray(xyz_scale)[:-1]

    # filter contours
    filt_area = (carea>cell_area_thresh[0]) & (carea<cell_area_thresh[1])
    circ = circularity(contours)
    filt_shape = (circ>cell_circularity_thresh)
    filt_intensity = mean_intensities >  np.percentile(im_orig, int(100*abs_intensity_percentile_thresh))
    filt = ((filt_area) & (filt_shape) & (filt_intensity))
    contours = contours[filt]

    centers = np.asarray([c.mean(axis=0) for c in contours])

    return centers, contours

def show_edges(im, edgeim, **kwargs):
    pl.imshow(im, interpolation='none', **kwargs)
    edgeim = edgeim.astype(float)
    edgeim[edgeim==0] = np.nan
    pl.imshow(edgeim, interpolation='none', vmax=1, **kwargs)

def cell_detect(im_orig, xyz_scale=(1,1,1), cell_area_thresh=[45,1850], cell_circularity_thresh=0.3, abs_intensity_percentile_thresh=0.2, edge_finding_param=0.99, sigma=2, dilation_kernel_size=2, showedgeim=False, returndisp=True): 
    '''Master Function for cell detection
    
    inputs:
        
    Returns:
        disp = image with contours
        centers = XY**** of centers (CV2 convention)
        contours = XY convention again
        
        
    '''
    if type(im_orig)==str:
        im_orig=tifffile.imread(im_orig)    
    im = process_image(im_orig)
    edge_im = find_edges(im, edge_finding_param=edge_finding_param, sigma=sigma, dilation_kernel_size=dilation_kernel_size)
    centers,contours=find_cells_intensity(edge_im, im_orig,  cell_area_thresh, cell_circularity_thresh, abs_intensity_percentile_thresh, xyz_scale=xyz_scale) #loosened up a bit
    #centers,contours=find_cells_intensity(edge_im, im_orig,  cell_area_thresh=[55,1750], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.9925, xyz_scale=xyz_scale) #origi
    if returndisp: 
        disp = display_cells(im_orig, contours)
        if showedgeim:
            return disp, centers, contours, edge_im
        else:
            return disp, centers, contours
    elif not returndisp:
        if showedgeim:
            return centers, contours, edge_im
        else:
            return centers, contours
def cell_detect_and_preprocess(im_orig, xyz_scale=(1,1,1), edge_finding_param=0.99, sigma=2, dilation_kernel_size=2):
    if type(im_orig)==str:
        im_orig=tifffile.imread(im_orig)
    im = process_image(im_orig)
    im = clearmap.preprocess(im)
    edge_im = find_edges(im, edge_finding_param=edge_finding_param, sigma=sigma, dilation_kernel_size=dilation_kernel_size)
    centers,contours=find_cells_intensity(edge_im, im_orig, cell_area_thresh=[230, 1100], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.2, xyz_scale=xyz_scale) #loosened up a bit
    #centers,contours=find_cells_intensity(edge_im, im_orig,  cell_area_thresh=[55,1750], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.9925, xyz_scale=xyz_scale) #origi
    disp = display_cells(im_orig, contours)
    return disp, centers, contours

#%%
'''
if __name__ == '__main__':

    fls=[os.path.join(fld, x) for x in os.listdir(fld)]; fls.sort()
    for i in fls:    
        disp, centers, contours=cell_detect_and_preprocess(i)
        sitk.Show(sitk.GetImageFromArray(disp))














    #find_cells(edge_im, im_orig, cell_area_thresh=[140,1450], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.01, xyz_scale=(1,1,1))    
    from skimage import exposure
    import SimpleITK as sitk    
    import pylab as pl
    path = '/jukebox/wang/pisano/PNI_Light_Sheet/Dextran_Test_AF488_AF647/160202_Dextran_Test_AF488_AF647_250msec_5micronz_dynamicfocus5steps_2sheets_13-54-33/13-54-33_Dextran_Test_AF488_AF647_250msec_5micronz_dynamicfocus5steps_2sheets_UltraII[01 x 01]_C500.ome.tif'
    path = '/Users/ben/Desktop/example.tif'
    path= '/home/wanglab/wang/pisano/tracing_output/vc22_lob6/full_sizedatafld/ch00/vc22_lob6_75msec_z3um_1hfds_0067na_2602sw_70per561_C00_Z0898.tif'
    path='/home/wanglab/wang/pisano/tracing_output/092915_crsi_02/full_sizedatafld/ch02/bl6_092915_crsi_02_w488_z3um_100msec_na0005_sw3862_1hfds_C02_Z0470.tif'

    im_orig = tifffile.imread(path).astype('uint16')
    claheim=clahe(im_orig.astype('uint16'), kernel_size=500)
    pl.hist(np.ravel(im_orig), bins=200)
    pl.figure()
    pl.hist(np.ravel(clahe), bins=200)
    
    #%%    
    def cell_detect2(im):
    from skimage.filters import threshold_adaptive
    im_th = threshold_adaptive(im, 55)
    
    def cell_detect3(im_orig, xyz_scale=(1,1,1)): 
    im = process_image(im_orig)
    im = clearmap.preprocess(im, bgrsize=9, DoGsize=8) #TP added
    im1=np.copy(im); im1[im1<14]=0
    edge_im = find_edges(im1, edge_finding_param=0.5, sigma=6, dilation_kernel_size=1)
    sitk.Show(sitk.GetImageFromArray(edge_im))
    centers, contours = find_cells(edge_im, im_orig, cell_area_thresh=[25,600], cell_circularity_thresh=0.5, abs_intensity_percentile_thresh=0.2, xyz_scale=xyz_scale)
    disp = display_cells(im_orig, contours)
    sitk.Show(sitk.GetImageFromArray(disp))
    return disp, centers, contours    
    
    
    
    #%%
    sitk.Show(sitk.GetImageFromArray(p))
    
    y,x=im_orig.shape

    im = process_image(im_orig)
    edge_im = find_edges(im, edge_finding_param=0.5, sigma=3, dilation_kernel_size=3)
    centers, contours = find_cells(edge_im, cell_area_thresh=[100,15000], cell_circularity_thresh=0.15)
    disp = display_cells(im, contours)
    color=np.zeros((1,y,x,3))
    
    color[:,:,:,1] = disp
    
    import cv2; cv2.fillPoly(im_orig, contours, color=(250,0,0))
    color[:,:,:,0] = im_orig
    sitk.Show(sitk.GetImageFromArray(color))
    disp, centers, contours=cell_detect(dataDoG, xyz_scale=(1.63,1.63,3))
    pl.hist(np.ravel(dataDoG), bins=2000)
    
    #%%using bilateral filter
    cm=clearmap.preprocess(im_orig)
    bf=cv2.adaptiveBilateralFilter(cm.astype(np.uint8), (3,3), 3); sitk.Show(sitk.GetImageFromArray(bf))
    edge_im = find_edges(im, edge_finding_param=0.85, sigma=2, dilation_kernel_size=2); sitk.Show(sitk.GetImageFromArray(edge_im))
    centers, contours = find_cells(edge_im, im_orig, cell_area_thresh=[10,3450], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.1, xyz_scale=xyz_scale)
    disp = display_cells(im_orig, contours); sitk.Show(sitk.GetImageFromArray(disp))    
    
    ##very good at getting rid of edges, can consider using a lax version of this with the BD's cell detect and count only if present in both?
    bf=cv2.adaptiveBilateralFilter(im_orig.astype(np.uint8), (3,3), 3); sitk.Show(sitk.GetImageFromArray(bf))
    cm=clearmap.preprocess(bf); sitk.Show(sitk.GetImageFromArray(cm))
    edge_im = find_edges(im, edge_finding_param=0.85, sigma=2, dilation_kernel_size=2); sitk.Show(sitk.GetImageFromArray(edge_im))
    centers, contours = find_cells(edge_im, im_orig, cell_area_thresh=[10,3450], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.01, xyz_scale=xyz_scale)
    disp = display_cells(im_orig, contours); sitk.Show(sitk.GetImageFromArray(disp))
   
    #%%using Nico's clearmap
    cm=clearmap.preprocess(im_orig)
    maxima=detect
    centers, contours = find_cells(edge_im, im_orig, cell_area_thresh=[10,3450], cell_circularity_thresh=0.4, abs_intensity_percentile_thresh=0.1, xyz_scale=xyz_scale)
    disp = display_cells(im_orig, contours); sitk.Show(sitk.GetImageFromArray(disp))    
   
'''
#%%

