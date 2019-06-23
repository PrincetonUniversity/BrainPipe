#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:46:46 2017

@author: tpisano
"""
from matplotlib import markers
from matplotlib.path import Path

def align_marker(marker, halign='center', valign='middle',):
    """ FROM https://stackoverflow.com/questions/26686722/align-matplotlib-scatter-marker-left-and-or-right
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    if isinstance(halign, (str, unicode)):
        halign = {'right': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'left': 1.,
                  }[halign]

    if isinstance(valign, (str, unicode)):
        valign = {'top': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'bottom': 1.,
                  }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)




def horizontal_plot_with_effect_size(df, experimental_columns, control_columns, pvalues=[0.00, 0.05]):
    '''Function to create a plot given a dataframe generated from 
    
    df = generated from function _________
    pvalues = range of pvalues to include
    experimental_columns: columns names for experimental group, e.g.
    control_columns
    
    '''
    from matplotlib import gridspec
    from tools.analysis.effectsize import cohen_d
    import matplotlib.pyplot as plt
    import os, numpy as np
    
    #inputs
    assert str(type(df)) == "<class 'pandas.core.frame.DataFrame'>"
    assert type(experimental_columns) == list
    assert type(control_columns) == list
    
    #make hbar plot grouped by structures of interest
    ndf=df[(df.pvalue>pvalues[0]) & (df.pvalue<pvalues[1])]
    
    progenitors = list(set(ndf.Progenitor.values))
    
    fig = plt.figure(figsize=(10, 5), dpi = 300, frameon=False)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
    ax = plt.subplot(gs[0])
    
    tick = 0
    progtick=0
    nnm = []
    plotted=[]
    effectsizes=[]
    #'sort' by sois progenitor areas
    for prog in progenitors:
        for idx, row in sss[sss.Progenitor==prog].iterrows():
            ctls = [row[xx] for xx in control_columns]
            stims = [row[xx] for xx in experimental_columns]
            effectsizes.append(np.round(cohen_d(stims, ctls), decimals=2))
            ax.scatter(y = [tick] * len(control_columns), x = ctls, color = 'red', marker = 'x', alpha=.9)
            ax.scatter(y = [tick] * len(experimental_columns), x = stims, facecolors='none', edgecolors = 'green', marker = 'D', alpha=1)
            nnm.append(row['name'])
            plotted.append((row['name'], prog, tick, progtick))
            tick+=1
        progtick+=1
    
    #plt.yticks(np.arange(0, len(nnm), 1.0))
    plt.yticks(range(len(nnm)), nnm)#, rotation = 40)
    ax.set_ylim(-.5, len(nnm)-.5)
    ax.set_xscale('log')
    ax.set_xlim(1, 150000)
    ax.set_xlabel('cFos positive cell counts')
    ax.set_ylabel('Structure')
    plt.title('Statistically significant structures')
    ax.legend()
    
    #add horizontal stripes and effect size
    for i in range(0, len(nnm)):
        plt.axhspan(i-.26, i+.26, facecolor='0.6', alpha=0.1)
        plt.text(160000, i-.12, effectsizes[i], size=9)
    
    #set cmap based on prog and change colors based on parent structure
    import seaborn as sns
    cmap = sns.husl_palette(len(progenitors), l=.6, s=.8, h=.1)
    del sns
    for p in plotted:
        plt.gca().get_yticklabels()[p[2]].set_color(cmap[p[3]]) 
        
    #legend
    ax = plt.subplot(gs[1])
    #ax = fig.add_subplot(1,2,2)
    ax.scatter(x = 0, y = 0, color = 'black', marker = 'o', alpha=1, label='Stimulation')
    ax.scatter(x = 0, y = 0, color = 'black', marker = 'x', alpha=1, label='Control')
    for prog in progenitors[::-1]:
        for p in set([(xx[1], xx[3]) for xx in plotted]):
            if prog == p[0]: ax.scatter(x = 0, y =0, color = cmap[p[1]], marker = 's', alpha=1, label=p[0])
    #for p in set([(xx[1], xx[3]) for xx in plotted]):
    #    ax.scatter(x = 0, y =0, color = cmap[p[1]], marker = 's', alpha=1, label=p[0])
    ax.set_ylim(0,1500000)
    ax.set_xlim(0,1500000)
    ax.patch.set_visible(False)
    ax.axis('off')
    ax.legend()
    
    plt.tight_layout()
    
