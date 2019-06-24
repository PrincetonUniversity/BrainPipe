import numpy as np, tifffile
import pylab as pl
from scipy.stats import linregress

im = tifffile.imread('brain.tif').astype(float)
ch1,ch2 = im.transpose([1,0,2,3])

ydata = ch1.flatten()
xdata = ch2.flatten()
good = (xdata>10) & (ydata>10)
idxs = np.arange(len(xdata))[good]
ydata,xdata = ydata[good],xdata[good]
ydata,xdata = map(np.log, [ydata,xdata])
m,yint,r,p,err = linregress(xdata,ydata)

yhat = m*xdata+yint
resid = ydata-yhat
dummyx = np.linspace(0,xdata.max(),20)
pl.plot(dummyx, m*dummyx+yint)
pl.scatter(xdata[::3000],ydata[::3000],c=resid[::3000])

ex = np.argwhere(resid>resid.mean()+5*resid.std()).squeeze()
ex_idxs = idxs[ex]

res = np.zeros(ch1.flatten().shape)
res[ex_idxs] = 1
res = res.reshape(ch1.shape)
