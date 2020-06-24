import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__': #old example with non updated percentages


    data = [('Brain',
              100,
              [('Telencephalon',
                55.062367663848136,
                [('Isocortex', 26.791080749679224, []),
                 ('Olfactory areas', 4.634975621772063, []),
                 ('Hippocampal formation', 9.335268514285922, []),
                 ('Cortical subplate', 2.235436409488298, []),
                 ('Striatum', 9.824827544649308, []),
                 ('Pallidum', 2.240778823973317, [])]),
               ('Diencephalon',
                7.9655087288508675,
                [('Thalamus', 4.294783085352911, []),
                 ('Hypothalamus', 3.6707256434979567, [])]),
               ('Mesencephalon', 8.104447240675304, [('Midbrain', 8.104447240675304, [])]),
               ('Metencephalon',
                13.222788533543628,
                [('Pons', 3.584074450030032, []), ('Cerebellum', 9.638714083513596, [])]),
               ('Myelencephalon', 7.011979314760745, [('Medulla', 7.011979314760745, [])]),
               ('Tracts', 8.63290851832132, [('fiber tracts', 8.63290851832132, [])])])]



    sunburst(data)
    sunburst(data, remove_labels=True)
    
    #basic color test
    color_lst = ['Brain','Telencephalon', 'Isocortex', 'Olfactory areas','Hippocampal formation','Cortical subplate',
                 'Striatum', 'Pallidum', 'Diencephalon','Thalamus','Hypothalamus','Mesencephalon','Midbrain','Metencephalon',
                 'Pons','Cerebellum','Myelencephalon', 'Medulla','Tracts', 'fiber tracts',]
    color_dct = {xx:'r' for xx in color_lst}
    sunburst_color(data, color_dct)
    
    #color test
    import seaborn as sns
    color_lst = ['Brain','Telencephalon','Diencephalon','Mesencephalon','Metencephalon','Myelencephalon', 'Tracts']
    color_dct = {xx:sns.color_palette("bone_r", 15)[i] for i,xx in enumerate(color_lst)}    
    colordf = pd.read_pickle('/jukebox/wang/pisano/figures/deformation_based_geometry/v3/data/colordf.p')
    color_dct.update({a:b for a,b in zip(colordf['soi'].tolist(), colordf['Color'].tolist())})
    sunburst_color(data, color_dct)
    


#%%
def sunburst(nodes, total=np.pi * 2, offset=0, level=0, ax=None, remove_labels=False):
    '''https://stackoverflow.com/questions/12926779/how-to-make-a-sunburst-plot-in-r-or-python
    '''
    ax = ax or plt.subplot(111, projection='polar')
    
    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        #if True:label = ''
        if not remove_labels: ax.text(0, 0, label, ha='center', va='center')
        if remove_labels: ax.text(0, 0, '', ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax, remove_labels=remove_labels)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,level=level + 1, ax=ax, remove_labels=remove_labels)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1, edgecolor='white', align='edge')
        
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            #rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            #print label, x,y, rotation, level
            if not remove_labels:ax.text(x, y, label, rotation=False, ha='center', va='center', size=(16 - (3*level))) #changed rotation to false
            if remove_labels:ax.text(x, y, '', rotation=False, ha='center', va='center', size=(16 - (3*level))) #changed rotation to false

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()
    return

def sunburst_color(nodes, color_dictionary, total=np.pi * 2, offset=0, level=0, ax=None, remove_labels=False):
    '''https://stackoverflow.com/questions/12926779/how-to-make-a-sunburst-plot-in-r-or-python
    '''
    ax = ax or plt.subplot(111, projection='polar')
    
    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        if not remove_labels: ax.text(0, 0, label, ha='center', va='center')
        if remove_labels: ax.text(0, 0, '', ha='center', va='center')
        sunburst_color(subnodes, color_dictionary, total=value, level=level + 1, ax=ax, remove_labels=remove_labels)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst_color(subnodes, color_dictionary, total=total, offset=local_offset,level=level + 1, ax=ax, remove_labels=remove_labels)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1, edgecolor='white', align='edge',color = [color_dictionary[xx] for xx in labels])
        
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            if not remove_labels:ax.text(x, y, label, rotation=False, ha='center', va='center', size=(16 - (3*level))) #changed rotation to false
            if remove_labels:ax.text(x, y, '', rotation=False, ha='center', va='center', size=(16 - (3*level))) #changed rotation to false

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()
    return