#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:25:05 2017

@author: tpisano
"""

'''

https://github.com/keplr-io/quiver********************************

List of resources: https://handong1587.github.io/deep_learning/2015/10/09/visulize-cnn.html

http://www.codesofinterest.com/2017/02/visualizing-model-structures-in-keras.html

https://github.com/raghakot/keras-vis/blob/master/vis/visualization/activation_maximization.py

https://github.com/yashk2810/Visualization-of-Convolutional-Layers/blob/master/Visualizing%20Filters%20Python3%20Theano%20Backend.ipynb
https://raghakot.github.io/keras-vis/vis.visualization/
https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb

https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py

https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer

https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer

https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4

https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

https://github.com/InFoCusp/tf_cnnvis/tree/master/examples

https://github.com/keras-team/keras/blob/master/examples/deep_dream.py

https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py

from Talmo
https://distill.pub/2017/feature-visualization/

Good for visualizations of different activation layers
https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
'''

#http://www.codesofinterest.com/2017/02/visualizing-model-structures-in-keras.html
from tools.conv_net.functions.buildModel import buildModelUnet3D, buildModelDetectionNet
from keras.utils.vis_utils import plot_model
model = buildModelUnet3D((49, 74, 74, 1))
plot_model(model, to_file='/home/wanglab/wang/pisano/conv_net/training/visualization/buildModelUnet3D_vert.png', show_shapes=False, show_layer_names=False)  
plot_model(model, to_file='/home/wanglab/wang/pisano/conv_net/training/visualization/buildModelUnet3D_hor.png', show_shapes=False, show_layer_names=False, rankdir='LR')  
model = buildModelDetectionNet((49, 74, 74, 1))
plot_model(model, to_file='/home/wanglab/wang/pisano/conv_net/training/visualization/buildModelDetectionNet_vert.png', show_shapes=False, show_layer_names=False)  
plot_model(model, to_file='/home/wanglab/wang/pisano/conv_net/training/visualization/buildModelDetectionNet_hor.png', show_shapes=False, show_layer_names=False, rankdir='LR')  
#%%
#quiver:https://github.com/keplr-io/quiver
#only seems to work for jpgs
from keras.models import load_model
from tools.conv_net.functions.customobjects import max_pred, min_pred, weightedCrossEntropy
from tools.utils.io import load_dictionary, makedir, listdirfull, load_np
from functools import partial, update_wrapper
threshold = 0.911588829975
src = '/home/wanglab/wang/pisano/conv_net/training/formatted_data/3d_jobid_0010/'
pth_to_model = '/home/wanglab/wang/pisano/conv_net/training/formatted_data/3d_jobid_0010/training_results/Weights_00_957729.0789_692093.0932.hdf5'


#load
preprocess_dictionary = load_dictionary(listdirfull(src, keyword = 'preprocess_dictionary.p')[0])
binary_cross_entropy_weight = preprocess_dictionary['binary_cross_entropy_weight'] if 'binary_cross_entropy_weight' in preprocess_dictionary else 50
binary_cross_entropy_func = update_wrapper(partial(weightedCrossEntropy, weight=binary_cross_entropy_weight), weightedCrossEntropy)
model = load_model(pth_to_model, custom_objects={'weightedCrossEntropy': binary_cross_entropy_func, 'max_pred' : max_pred, 'min_pred' : min_pred})


#data
data = {xx: '{}_{}.npy.npz'.format(os.path.join(src, 'formatted_data'), xx) for xx in ['hasCellsInShuffledTest', 'hasCellsInShuffledTrain', 'hasCellsInShuffledVal', 'hasCellsOutShuffledTest', 'hasCellsOutShuffledTrain', 'hasCellsOutShuffledVal']}
testdata = np.asarray([norm(xx, mean_subtract=True) for xx in load_np(data['hasCellsInShuffledTest'])])
temp_folder='/home/wanglab/wang/pisano/conv_net/training/visualization/quiver'
makedir(temp_folder)
input_folder = os.path.join(temp_folder, 'input_images'); makedir(input_folder)
for i in range(len(testdata)):
    tifffile.imsave(input_folder+'/{}'.format(i), np.expand_dims(testdata[i,:,:,:,:], axis=0).astype('float32')) #might need to change to float32


from quiver_engine import server
server.launch(
    model, # a Keras Model
    # where to store temporary files generatedby quiver (e.g. image files of layers)
    temp_folder=temp_folder, 
    # a folder where input images are stored
    input_folder=input_folder,
    # the localhost port the dashboard is to be served on
    port=5000)

