import os,sys, pickle
import numpy as np 
#ClearMap path
#sys.path.append('/usr/people/ejdennis/.local/bin')

sys.path.append('../ClearMap2')
#load ClearMap modules
#from ClearMap.Environment import *  #analysis:ignore

print('about to import workspace')
import ClearMap.IO.Workspace as wsp
import ClearMap.IO.IO as io
import ClearMap.ImageProcessing.Experts.Cells as cells
import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.Alignment.Resampling as res

cell_detection_parameter = cells.default_cell_detection_parameter.copy()
cell_detection_parameter['illumination'] = None
cell_detection_parameter['maxima_detection']['valid']=True # keep this on if you want to combine results from multiple blocks

cell_detection_parameter['background_correction']['save'] = None
cell_detection_parameter['intensity_detection']['measure'] = ['source','background']
cell_detection_parameter['verbose'] = False # I set this to False because I didn't want to see the output like "running background removal" stuff for each of the ~100 or so blocks

cell_detection_parameter['maxima_detection']['save'] = None

def process_block(block,params=cell_detection_parameter,):
    """
    written by Austin Hoag, "borrowed" by Emily Dennis
    ---PURPOSE---
    A function that takes a block as input and 
    runs the cells.detect_cells_block() function on it.
    
    We then save the results in an array so that we 
    can just load them later when we want to merge this all together
    ---INPUT---
    block                       A processing block created from bp.split_into_blocks()
    cell_detection_parameters   The cell detection parameter dictionary 
                                that you feed into detect_cells_block()
    ---OUTPUT---
    block_result      The tuple containing the cell coordinates, shape, intensities 
    It also saves this block_result as a file in your output directory called:
                      "cells_block{block_index}.p" where block_index is ranges from 0 to the number of blocks-1
    """
    block_index = block.index[-1]
    block_result = cells.detect_cells_block(block, parameter=params)
    block_savename = os.path.join(directory,'final_blocks',f'cells_block{block_index}.p')
    with open(block_savename,'wb') as pkl:
        pickle.dump(block_result,pkl)
    print(f"Saved {block_savename}")
    return block_result

if __name__ == '__main__':

	print('starting')
	#directories and files
	jobid = int(os.environ["SLURM_ARRAY_TASK_ID"])
	step = int(sys.argv[1])
	print('sysarg v output is {}'.format(sys.argv))

	if sys.argv[3] == "lavision":
		cell_detection_parameter['background_correction']['shape'] = (3,3)
		cell_detection_parameter['shape_detection']['threshold'] = 300
		print('/n params are k3, 300')
	else:
		cell_detection_parameter['background_correction']['shape'] = (5,5)
		cell_detection_parameter['shape_detection']['threshold'] = 130
		print('/n params are k5, 130')

	directory = str(sys.argv[2]) #e.g. os.path.join('/scratch/ejdennis/cm2_brains/j317/ch_488/')

	expression_raw      = 'Z<Z,4>.tif'    
	expression_auto	= 'Z<Z,4>.tif'

	ws = wsp.Workspace('CellMap', directory=directory);
	ws.update(raw=expression_raw)
	#ws.debug = 'medchunk'	
	print(ws.info())

	# convert raw to stitched npy file      
	source = ws.source('raw');
	sink   = ws.filename('stitched')

	if not os.path.exists(os.path.join(directory,"final_blocks")):
		os.mkdir(os.path.join(directory,"final_blocks"))	

	if step == 0:
		print("++++++++++ STEP 0 +++++++++++++")
		# convert single z planes to stitched
		io.delete_file(sink)
		io.convert(source, sink, processes=None, verbose=True);

	elif step == 1:
		# Split into blocks
		print("splitting into blocks")
		blocks = bp.split_into_blocks(ws.source('stitched'), 
			processes=12, 
			axes=[2], # chunks along z
			size_min=5,
			size_max=20, 
			overlap=2,
			verbose=True)
		print("Done splitting into blocks")
		
		# run cell detection on each block
		print(ws.info())
		block = blocks[jobid]
		print(f"Running cell detection on single block: blocks[{jobid}]")
		sys.stdout.flush()
		block_result = process_block(block,params=cell_detection_parameter)
		print("Done running cell detection")
	
	elif step == 3:
		# merge blocks
		list_of_blocks = os.listdir(os.path.join(directory,'final_blocks'))
		block_result_list = []
		for block_file_base in list_of_blocks:
			block_file = os.path.join(directory,'final_blocks',block_file_base)
			with open(block_file,'rb') as file_to_load:
				block_result = pickle.load(file_to_load)
				block_result_list.append(block_result)
		final_results = np.vstack([np.hstack(r) for r in block_result_list]) # merges results into a single array
		header = ['x','y','z'];
		dtypes = [int, int, int];
		if cell_detection_parameter['shape_detection'] is not None:
			header += ['size'];
			dtypes += [int];
		measures = cell_detection_parameter['intensity_detection']['measure'];
		header +=  measures
		dtypes += [float] * len(measures)

		dt = {'names' : header, 'formats' : dtypes};
		cells_out = np.zeros(len(final_results), dtype=dt);
		for i,h in enumerate(header):
			cells_out[h] = final_results[:,i];
		ws.filename('cells',postfix='raw')
		savename = ws.filename('cells',postfix='raw')
		io.write(savename,cells_out)
