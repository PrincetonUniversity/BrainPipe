# Analysis scripts for light sheet microscopy for the Brody lab.

Edits made by Emily Jane Dennis (ejdennis@princeton) with significant help from Zahra (zmd@princeton). Forked from PrincetonUniversity/BrainPipe by Tom Pisano, Zahra John D'Uva. It also includes modified scripts from ClearMapCluster and lightsheet_helper_scripts written by the same people. ClearMap Cluster is T. Pisano's parallelization to a cluster of C. Kirst's ClearMap software (https://idisco.info/clearmap/) for use on a cluster using a slurm based scheduler. Written for Python 3.7+. Modifications by Zahra M.  This is using ClearMap 1, but ClearMap2 is now available and we should move to that reasonably soon.

Includes three-dimensional CNN with a U-Net architecture (Gornet et al., 2019; K. Lee, Zung, Li, Jain, & Sebastian Seung, 2017) with added packages developed by Kisuk Lee (Massachusetts Institute of Technology), Nick Turner (Princeton University), James Gornet (Columbia University), and Kannan Umadevi Venkatarju (Cold Spring Harbor Laboratories).

# This is a work in progress. Emily has largely updated scripts and README, but some documentation is not fully fleshed out.

### *Dependencies:*
[DataProvider3](https://github.com/torms3/DataProvider3)  
[PyTorchUtils](https://github.com/nicholasturner1/PyTorchUtils)  
[Augmentor](https://github.com/torms3/Augmentor)  
[DataTools](https://github.com/torms3/DataTools)

## *INSTALLATION INSTRUCTIONS*:
* Note that this currently has only been tested on Linux (Ubuntu 16 and 18).
* If on a cluster - Elastix needs to be compiled on the cluster - this was challenging for IT here and suspect it will be for your IT as well. If at Princeton, elastix is on spock

### Create an anaconda python environment
 - use [anaconda](https://www.anaconda.com/download/)
- name an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) 'lightsheet' (in python 3.7+)

```
$ conda create -n lightsheet python=3.7.3
$ pip install cython futures h5py joblib matplotlib natsort numba numpy opencv-python openpyxl pandas scipy scikit-image scikit-learn seaborn SimpleITK tifffile tensorboardX torch torchvision tqdm xlrd xvfbwrapper
```

If on a local Ubuntu machine also install elastix and xvfb, and make sure you have all boost libraries installed for DataTools:

```
$ sudo apt-get install elastix
$ sudo apt-get install xvfb
$ sudo apt-get install libboost-all-dev
```

[Download](https://github.com/abria/TeraStitcher/wiki/Binary-packages) and unpack TeraStitcher
```
$ bash TeraStitcher-Qt4-standalone-1.10.16-Linux.sh?dl=1
```
* Modify Path in ~/.bashrc:

```
export PATH="<path/to/software>TeraStitcher-Qt4-standalone-1.16.11-Linux/bin:$PATH"
```
* Check to see if successful

```
$ which terastitcher
```


Navigate to `tools/conv_net` and clone the necessary C++ extension scripts for working with DataProvider3:

```
$ git clone https://github.com/torms3/DataTools.git
```

Go to the dataprovider3, DataTools, and augmentor directories in `tools/conv_net` and run (for each directory):

```
$ cd tools/conv_net/dataprovider3
$ python setup.py install
$ cd ../DataTools
$ python setup.py install
$ cd ../augmentor
$ python setup.py install

```

### If not at Princeton - make sure your slurm scheduler works similarly
* If not, change .sh in main folder, /slurm_scripts in the main rat_BrainPipe, and ClearMapCluster folders like sub_main_tracing.sh file, e.g.
```
module load anacondapy/5.3.1
module load elastix/4.8
. activate ligthsheet
```

* Check/change the resource allocations and email alerts at the top of each .sh file based on cluster and run_tracing.py settings

### Edit: lightsheet/tools/utils/directorydeterminer:
* Add your paths for BOTH the cluster and local machinery

### Edit: ADD LIST OF FILES TO CHANGE

## To train a CNN to identify features (like somas)

0. Make test sets
    Emily made a document with explicit instructions and examples
    which can be found [here](https://docs.google.com/document/d/1f5owGhyJiL2dNqIMZ_zwM96QTBm2_WKqoY5HwBRAMVA/edit)

1. Find parameters to use
2. Train network
3. Validate and Benchmark Network
    - Detailed instructions on steps 1-3 can be found [here](https://docs.google.com/document/d/1cuNthPY2Z-69SQi9aSwfbgJlHpvQGivhxFtOUmAKOm4/edit#)

4. Run inference on whole brains

5. Visualize network outputs

6. To compare to ClearMap
    See next steps on how to run clearmap, visualize, and compare ouputs

**Note** there is a demo for the CNN to test if your paths are correct and everything is installed properly, see end of document

## To use ClearMap to identify cell centers

0. Make test sets
    If you have already made test sets to train a CNN, use those.

1. Find parameters
- See `ClearMapCluster/ parameter_sweep.ipynb`

2. Run ClearMap on whole brains

3. Visualize outputs

4. Compare to CNN (if desired)
    - on training data
    - on whole brains

## Using raw lightsheet images to:

### 1. Make a stitched, whole-brain

### 2. Make an atlas

### 3. Put a brain in atlas space
 - visualize warping procedure

### 4. Add annotations to an atlas




_______________________
# From old readme:

## To run, I suggest:
* Open `run_tracing.py`
* For **each** brain modify:
	* `inputdictionary`
	* `params`
	* **NOTE** we've noticed that elastix (registration software) can have issues if there are spaces in path name. I suggest removing ALL spaces in paths.
* Then, I suggest, using a local machine, run 'step 0' (be sure that `run_tracing.py` is edited **before**):

* **why**: This generates a folder where data will be generated, allowing to run multiple brains on the cluster at once.
* then using the cluster's headnode (in the **new** folder's lightsheet directory generated from the previous step) submit the batch job: `sbatch sub_registration.sh`

# *Descriptions of important files*:
- main GPU-based scripts are located in the pytorchutils directory
1. `run_exp.py` --> training
    - lines 64-98: modify data directory, train and validation sets, and named experiment   	  directory (in which the experiment directory of logs and model weights is stored)
2. `run_fwd.py` --> inference
    - lines 57 & 65: modify experiment and data directory
3. `run_chnk_fwd.py` --> large-scale inference
    - lines 82 & 90: modify experiment and data directory
    - if working with a slurm-based scheduler:
	1. modify `run_chnk_fwd.sh` in `pytorchutils/slurm_scripts`
	2. use `python pytorchutils/run_chnk_fwd.py -h` for more info on command line 		arguments
4. modify parameters (stride, window, # of iterations, etc.) in the main parameter dictionaries
- `cell_detect.py` --> CPU-based pre-processing and post-processing
	- output is a "3dunet_output" directory containing a '[brain_name]_cell_measures.csv'
    - if working with a slurm-based scheduler,
	1. `cnn_preprocess.sh` --> chunks full sized data from working processed directory  
	2. `cnn_postprocess.sh` --> reconstructs and uses connected components to find cell measures

* `sub_registration.sh` or `sub_registration_terastitcher.sh`:
	* `.sh` file to be used to submit to a slurm scheduler
	* this can change depending on scheduler+cluster but generally batch structure requires 2 variables to pass to `run_tracing.py`:
		* `stepid` = controlling which 'step' to run
		* `jobid` = controlling which the jobid (iteration) of each step
	* Steps:
		* `0`: set up dictionary and save; requires a single job (jobid=0)
		* `1`: process (stitch, resize) zplns, ensure that 1000 > zplns/slurmfactor. typically submit 80 jobs for LBVT (jobid=0-80).
		* `2`: resample and combine; typically submit 3 jobs (requires 1 job/channel; jobid=0-3)
		* `3`: registration via elastix

* `sub_main_tracing.sh`:
	* `.sh` file to be used to submit to a slurm scheduler
	* this can change depending on scheduler+cluster but generally batch structure requires 2 variables to pass to `run_tracing.py` AND `cell_detect.py`:
		* `stepid` = controlling which 'step' to run
		* `jobid` = controlling which the jobid (iteration) of each step
	* Steps:
		* `0`: set up dictionary and save; requires a single job (jobid=0)
		* `1`: process (stitch, resize) zplns, ensure that 1000 > zplns/slurmfactor. typically submit 80 jobs for LBVT (jobid=0-80).
		* `2`: resample and combine; typically submit 3 jobs (requires 1 job/channel; jobid=0-3)
		* `3`: registration via elastix
		* `cnn_preprocess.sh` (will add to this)

* `run_tracing.py`:
	* `.py` file to be used to manage the parallelization to a SLURM cluster
	* inputdictionary and params need to be changed for each brain
	* the function `directorydeterminer` in `tools/utils` *REQUIRES MODIFICATION* for both your local machine and cluster. This function handles different paths to the same file server.
	* generally the process is using a local machine, run step 0 (be sure that files are saved *BEFORE( running this step) to generate a folder where data will be stored
	* then using the cluster's headnode (in the new folder's lightsheet directory generated from the previous step) submit the batch job: `sbatch sub_registration.sh`

* `cell_detect.py`:
	* `.py` file to be used to manage the parallelization _of CNN preprocessing_ to a SLURM cluster
	* params need to be changed per cohort.
	* see the tutorial for more info.

* tools: convert 3D STP stack to 2D representation based on colouring
  * imageprocessing:
	* `preprocessing.py`: functions use to preprocess, stitch, 2d cell detect, and save light sheet images
  * analysis:
	* `allen_structure_json_to_pandas.py`: simple function used to generate atlas list of structures in coordinate space
	* other functions useful when comparing multiple brains that have been processed using the pipeline

* supp_files:
  * `gridlines.tif`, image used to generate registration visualization
  * `allen_id_table.xlsx`, list of structures from Allen Brain Atlas used to determine anatomical correspondence of xyz location.

* parameterfolder:
  * folder consisting of elastix parameter files with prefixes `Order<#>_` to specify application order


  # CNN Demo:
  - demo script to run training and large-scale inference
  - useful to make sure the environment and modules are imported correctly

  1. if working with a slurm-based scheduler:
  	1. run `sbatch run_demo.sh` within the tools/conv_net
  		* make sure you have "lightsheet" environment set up before running

  2. else, navigate to tools/conv_net; in the terminal, in the lightsheet environment, run:
  ```
  $ python setup_demo_script.py
  $ cd pytorchutils/
  $ python demo.py demo models/RSUNet.py samplers/demo_sampler.py augmentors/flip_rotate.py 10 --batch_sz 1 --nobn --noeval --tag demo
  ```
  3. output will be in a 'tools/conv_net/demo/cnn_output' subfolder (as a TIFF)
