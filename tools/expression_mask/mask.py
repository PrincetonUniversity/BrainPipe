import numpy as np, os, cv2, sys
from scipy.stats import linregress
import multiprocessing as mp
from matplotlib import pyplot as pl
from scipy.cluster.vq import kmeans2
import tifffile
pl.ioff()

#%%
if __name__ == '__main__':
    
    src = '/jukebox/wang/abadura/TissueVision/ben/brains'
    dst = '/jukebox/wang/abadura/TissueVision/ben/brains'
    jobid = int(sys.argv[1])
    mask_arrayjob(jobid, src, dst, signal_index=0, regression_index=1)
    #add shutil to delete src
    
    
    
#%%                
def mask_arrayjob(jobid, src, dst, signal_index=0, regression_index=1):
    '''built for array jobs
    '''

    fs = [os.path.join(src,f) for f in os.listdir(src) if f.endswith('.tif')]
    if jobid >= len(fs):
        print('Job id greater than number of files.')
        sys.exit(0)

    print(fs[jobid])
        
    try:
        make_mask(fs[jobid], ch_idxs=(signal_index, regression_index), out=dst)
        print('Success for job {} : {}'.format(jobid,fs[jobid]))
    except:
        print('Error generating mask for job {} : {}'.format(jobid, fs[jobid]))
        raise

    return

#%%
def regress(c1_, c2_, mask, step):
    c1 = c1_[mask]
    c2 = c2_[mask]
    m,yint,_,_,_ = linregress(c1,c2)
    res = m*c1+yint - c2
    pos,neg = res>0, res<0
    pc_pos = np.percentile(np.abs(res[pos]), 100.-step)
    pc_neg = np.percentile(np.abs(res[neg]), 100.-step)
    pc_dif = pc_pos-pc_neg
    return m,yint,pc_dif,pos,neg,res,pc_pos,pc_neg



#%%
def make_mask(sigtif, regtif, step=0.05, slope_thresh=0.4, init_window=300, out='/jukebox/wang/abadura/TissueVision/ben/brains', despeckle_kernel=5, imsave_kwargs=dict(compress=5), save_plots=True, verbose=True, **kwargs):
    """
    Parameters
    ----------
    sigtif : str
        path to signal tiff file with ordering [z, y, x]
    regtif : str
        path to reg tiff file with ordering [z, y, x]
    step : float
        regression parameter, determine the step size to take when iteratively regressing
    slope_thresh : float
        threshold for slope of regression line, used both to determine initial regression mask, and specifies breakpoint for iterations    
    init_window : int
        size of window to step along when initializing regression: PREVIOUSLY AT 100 CHANGED TO 300 20170204 TP
    out : str
        path to output directory
    despeckle_kernel : odd integer
        kernel for median filter to despeckle mask
    imsave_kwargs : dict
        parameters for tifffile.imsave
    save_plots : bool
        save images of regression plots for inspection
    verbose : bool
        show status while running

    Returns and saves: mask of the same shape as one channel of tif, where each pixel value represents the pseudo-significance of that pixel (value of its brightness in units of standard deviations from inferred standard regression line between signal and background channel). Negative values and values masked by the regression process are zeroed out. The 'cloud.npy' file contains the indices of the flattened tif that represent the symmetrical point cloud around the regression; i.e. pixels that are not signal
    
    
    MODIFIED BY TP 9/8/16
    """

    name = kwargs['outputdirectory'][kwargs['outputdirectory'].rfind('/')+1:]
    if verbose: sys.stdout.write('Generating mask for {}'.format(name)); sys.stdout.flush()
    
    #naming
    mask_filename = name + '_mask.tif'
    cloud_filename = name + '_cloud.npy'
    plot_filename = name + '_plot.png'

    #handle pths
    if type(sigtif) == str:
        sigtif = tifffile.imread(sigtif)
    if type(regtif) == str:
        regtif = tifffile.imread(regtif)

    im = np.zeros((2, sigtif.shape[0], sigtif.shape[1], sigtif.shape[2]))
    im[0,...]=sigtif
    im[1,...]=regtif
    im = np.swapaxes(im, 0, 1).astype('uint16')
    del sigtif; del regtif
    #bd's stuff
    chs = im.transpose([1,0,2,3]); del im
    cs = [i.ravel() for i in chs]

    ch_idxs = (0,1)

    # pull out specified signal and background channels, var naming convention: 1=signal, 2=background
    c1,ch1 = cs[ch_idxs[0]],chs[ch_idxs[0]]
    c2,ch2 = cs[ch_idxs[1]],chs[ch_idxs[1]]
    del cs; del chs
    # remove major outliers
    outliers = c1>c1.mean()+15*c1.std()
    #c1[outliers] = 0
    #c2[outliers] = 0
    del outliers
    # generate initial mask and regression
    i,m,win = 0,0,init_window
    while m<slope_thresh:
        m,_,_,_,_ = linregress(c1[(c1>=i*win) & (c1<(i+1)*win)], c2[(c1>=i*win) & (c1<(i+1)*win)])
        i += 1
    mask = (c1>=i*win)
    m,yint,dif,pos,neg,res,pc_pos,pc_neg = regress(c1,c2,mask,step=step)

    # regression iterations
    i = 0
    while True:
        last_mask = mask.copy()
        last_par = [m,yint]
        if dif < 0 and m > slope_thresh:
            break
        exclude_idxs = np.arange(len(mask))[mask][res > pc_pos]
        mask[exclude_idxs] = False
        m,yint,dif,pos,neg,res,pc_pos,pc_neg = regress(c1,c2,mask,step=step)
        i += 1

        if verbose:
            sys.stdout.write('\n   {}, {}, {}'.format(i, dif, m)); sys.stdout.flush()
        ##TP ADDING to stop runaway regressions
        if i > 1000:
            sys.stdout.write('\n **STOPPPING REGRESSION AFTER 1000 ITERATIONS**\n\n'); sys.stdout.flush()            
            break

    mask = last_mask
    m,yint = last_par

    maskresid = m*c1[mask]+yint - c2[mask]
    residstd = np.std(maskresid)

    resid = (m*c1+yint - c2)
    symmetry_cloud = ((resid<0) | (mask))
    expr = resid / residstd # expression normalized to std of residuals in this brain
    expr[symmetry_cloud] = 0

    mask_im = np.zeros(ch1.shape, dtype=np.float32)
    mask_im.flat[expr>0] = expr[expr>0]
    mask_im = np.array([cv2.medianBlur(i,despeckle_kernel) for i in mask_im], dtype=np.float32)
    mask_im = mask_im[:,None,...]
    
    if save_plots:
        fig,axs = pl.subplots(2,1)
        axs = axs.ravel()
        axs[0].scatter(c1[::100], c2[::100], marker='x', c=expr[::100], s=1./(mask.astype(int)[::100]+1)**2)
        x = np.array([c1[mask].min(), np.percentile(c1[mask], 99)])
        axs[0].plot(x, m*x+yint, color='k')
        imsh = axs[1].imshow(mask_im[:,0,...].max(axis=0), vmax=80.0)
        fig.colorbar(imsh)
        pl.savefig(os.path.join(out, plot_filename))

    mask_im = np.squeeze(mask_im)    
    
    tifffile.imsave(os.path.join(out, mask_filename), mask_im, **imsave_kwargs)
    np.save(os.path.join(out, cloud_filename), np.argwhere(symmetry_cloud))

    
    return mask_im
