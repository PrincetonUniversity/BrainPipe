# # Create_PMA_ontology_JSON.py
# @author: Austin Hoag
# @date: 2020-09-17
# The purpose of this script is to create the ontology JSON file for the Princeton Mouse Atlas,
# just like what Allen provides for the AMBA.
# We will start from a CSV file containing the regions names, ids, and their parent:child relationships.
#
#  
# ### Required:
# - python >= 3.6
# - numpy
# - pandas 

import json
import numpy as np
import pandas as pd


# Read in csv file into pandas dataframe
df_filename = '/jukebox/LightSheetTransfer/atlas/Princeton_mouse_atlas_id_table_public.csv'
df_pma = pd.read_csv(df_filename)

# Only keep necessary columns from the pandas dataframe 
df_pma = df_pma[['name','acronym','id','parent_structure_id']]
df_pma

"""
Make a massive dictionary where keys are ids and values are dictionaries containing the
info we will need in the ontology JSON file
We will initialize each entry with an empty 'children' key
which we will fill using the recursive function below
"""
pma_df_dict = {}
for index, (name,acronym,ID,parentid) in df_pma.iterrows():
    if name != 'root': # we will handle the root dictionary separately below
        pma_df_dict[ID] = {'id':int(ID),'name':name,'acronym':acronym,
                           'parent_structure_id':int(parentid),'graph_order':0,'children':[]}

# intialize the graph_order global variable that we will increment as we add the children
graph_order = 0 

def make_ontology_dict(dic):
    """
    ---PURPOSE---
    A recursive function that completes a
    dictionary of the same format as the allen.json file,
    using the pma_df_dict we just made.
    ---INPUT---
    dic:    A parent dictionary to which you want to add child dictionaries
    """
    # Find all children of the current ID
    parent_id = dic['id']
    child_ids = [v['id'] for v in pma_df_dict.values() if v['parent_structure_id'] == parent_id]
    # add the correct graph_order value to this id
    global graph_order
    dic['graph_order'] = graph_order
    graph_order+=1 # increment so the next in line in the hierarchy has a higher graph_order
    """ Make list of child dicts to add to the entry in the ontology dictionary"""
    child_dicts = [pma_df_dict[ID] for ID in child_ids]
    dic['children']=child_dicts
    for child in child_dicts: 
        child_id = child['id']
        make_ontology_dict(child) # recursive call - child is a dictionary itself
    return

# Initialize the final dictionary which will contain the nested structure with the root id 
ontology_dic = {'id':997,'name':'root','graph_order':0,'parent_structure_id':None,'children':[]}

# Call the recursive function, using the intialized root dictionary as the starting point
# The function fills out the rest of the dictionary
make_ontology_dict(dic=ontology_dic)

# Now write this to a JSON file
PMA_json_filename = './PMA_ontology.json'
with open(PMA_json_filename,'w') as outfile:
    json.dump(ontology_dic,outfile,indent=2) # indent is used to make formatting identical to allen.json
print("Wrote ontology dictionary to: ", PMA_json_filename)