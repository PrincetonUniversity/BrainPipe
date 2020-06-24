import numpy as np


def get_target_array(input_array,centroids,distance_fn,A = 3.0/5.0):
	"""input_array is a 3D array of raw activation values.
	centroids is a list of 3 component vectors. A is the
	scaling factor (number of microns per z pixel over number
	of microns per x pixel). Ordering is assumed to be x,y,z"""
	dims = input_array.shape
	arrs = []
	for dimension in dims:
		arrs.append(1.0*np.arange(0,dimension))
	meshes = np.meshgrid(*arrs,indexing='ij')
	output_array = np.zeros_like(input_array)
	for c in centroids:
		dd = get_distance_array(meshes,c,A)
		output_array = np.maximum(output_array,distance_fn(dd))
	return output_array

def distance_function_exp(arr):
	scale = 3.0 #scale in pixels
	return np.exp(-arr/scale)


def get_distance_array(meshes,c,A):
	assert(len(meshes) == len(c))
	dd = np.zeros_like(meshes[0])
	for i,mm in enumerate(meshes):
		if i > 0:
                 dd += (mm - c[i])**2
		else: #apply A factor on the last one
                 dd += (A*(mm - c[i]))**2
	dd = dd**0.5
	return dd

def apply_dilation(src, A = 3.0/5.0):
    '''wrapper function for compatibility with previously made code - TP
    '''
    #src is a binary array where values >0 are centroids
    centroids=np.asarray(np.nonzero(src)).T

    return get_target_array(src,centroids,distance_function_exp, A)

def generate_target_array_selem(A = 3.0/5.0):
    '''To increase efficiency - making the element once computationally and then applying it.
    
    See tools/conv_net/functions/dilation
    '''

    return get_target_array(np.zeros((31,31,31)),[np.array([15,15,15])],distance_function_exp, A)
    

if __name__ == '__main__':
	input_array = np.random.randn(150,120, 100)
	centroids = [np.array([40,40,40]),np.array([80,70,60])]
	# input_array = np.random.randn(100,100,100) #example 3D data
	# centroids = [np.array([40,40,40]),np.array([80,70,75])]
	distance_fn = distance_function_exp
	output_array = get_target_array(input_array,centroids,distance_fn)

	#plot results
	import matplotlib.pyplot as plt
	plt.ioff()
	plt.imshow(np.max(input_array, axis=0))
	plt.figure()
	plt.imshow(np.max(output_array, axis=0))
	plt.show()

    #
	src = np.zeros((100,100, 100))
	src[70,70,70] = 1
	src[50,50,50] = 1
	plt.imshow(np.max(apply_dilation(src), axis=0))
    
	#
	src = generate_target_array_selem(A = 3.0/5.0)
	plt.imshow(np.max(src, 0))
    

