#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:18:52 2018

@author: tpisano
sudo apt-get install python-vtk
conda create --name mayavi python=3 scikit-learn pandas numpy scikit-image
pip install mayavi 
pip install PyQt5
source activate mayavi

"""
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from skimage import measure
from scipy.ndimage import zoom
from plotly.graph_objs import Figure

def vol_render(vol, level=0, color='r', opacity=0.5, save = False, zoom_factor=False, name = "Isosurface"):
    '''modified from #https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil
    
    Inputs
    ------
    vol == 
    level : float (from measure.marching_cubes)
        Contour value to search for isosurfaces in `volume`. If not
        given or None, the average of the min and max of vol is used.
        
    
    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    '''
    
    if zoom_factor: vol = zoom(vol, zoom_factor, order=1)
    
    if color=='r': colormap=((.8,.1,.1),(.8,.1,.1))
    #if color=='g': colormap=((.1,.8,.1),(.1,.8,.1))
    if color=='b': colormap=((.1,.1,.8),(.1,.1,.8))
    if color=='bb': colormap=((.6,.3,.8),(.5,.25,.8))
    if color=='o': colormap=[(0.4, 0.15, 0), (1, 0.65, 0.12)]
    if color=='g': colormap=[(0.15, 0.4,0), (0.65, 1, 0.12)]
    if color=='p': colormap=[(0.15, 0,0.4), (0.65, 0.12,1)]
    
    
    vertices, faces, normals, values = measure.marching_cubes(vol, 0)
    x,y,z = zip(*vertices)  
    fig = ff.create_trisurf(x=x,y=y, z=z, plot_edges=False,colormap=colormap,simplices=faces,title=name)
    
    #opacity
    fig['data'][0].update(opacity=opacity)
    
    #orientation
    #fig['layout'].update(dict(scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)))))
    fig['layout'].update(dict(scene=dict(camera=dict(eye=dict(x=1.25, y=.25, z=.25)))))
    plotly.offline.plot(fig)
    
    
def vols_render(vols, colors=False, opacity=0.5, save = False, zoom_factor=False, name = "Isosurface"):
    '''modified from #https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-in-matplotlib-mplot3d-or-simil
    
    
    LOOK INTO https://plot.ly/~empet/14613/isosurface-in-volumetric-data/#/
    
    Inputs
    ------
    vols ==  LIST MULITPLE
    level : float (from measure.marching_cubes)
        Contour value to search for isosurfaces in `volume`. If not
        given or None, the average of the min and max of vol is used.
        
    
    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    '''
    if not colors: colors = [#((.8,.1,.1),(.8,.1,.1)),
                              #((.1,.1,.8),(.1,.1,.8)),
                              ((.6,.3,.8),(.5,.25,.8)),
                            ((0.4, 0.15, 0), (1, 0.65, 0.12)),
                            ((0.15, 0.4,0), (0.65, 1, 0.12)),
                            ((0.15, 0,0.4), (0.65, 0.12,1))]
    
    colors = ['rgb(255,105,180)', 'rgb(255,255,51)']
    
    xx=[]; yy=[];zz=[]; faceslst=[]; out=[]
    for ii,vol in enumerate(vols):
        if zoom_factor: vol = zoom(vol, zoom_factor, order=1)        
        color = colors[ii]
        
        vertices, faces, normals, values = measure.marching_cubes(vol, 0)
        x,y,z = zip(*vertices)  
        print(faces.shape)
        
        for i in range(len(x)):
            xx.append(x[i])
            yy.append(y[i])
            zz.append(z[i])
        for i in range(len(faces)):
            faceslst.append(faces[i])
            out.append(color)
            
        
    fig = ff.create_trisurf(x=tuple(xx),y=tuple(yy),z=tuple(zz), plot_edges=False,colormap=out,simplices=np.asarray(faceslst),title=name)
            
    #opacity
    fig['data'][0].update(opacity=opacity)
    
    #orientation
    fig['layout'].update(dict(scene=dict(camera=dict(eye=dict(x=1.25, y=.25, z=.25)))))
        
    plotly.offline.plot(fig)
    
    return


layout = go.Layout(
    title='Another Parametric Plot',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        )
    )
)


#sudo apt-get install libvtk5-dev python-vtk
#LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/vtk-5.6"
#PYTHONPATH="$PYTHONPATH:/usr/local/lib/vtk-5.6"

if __name__ == '__main__':
    from skimage.morphology import ball
    vol = ball(10)
    
    vol = tifffile.imread('/home/wanglab/wang/pisano/figures/deformation_based_geometry/v2/registration/hp_pc/hann.tif')
    vol = np.flip(np.flip(np.swapaxes(zoom(vol, 0.20, order=1),0,1),0),2)
    
    save = '/home/wanglab/Downloads/tmp.svg'
    
    #
    from skimage.morphology import ball
    vols = [ball(3), ball(5)]
    vols_render(vols)