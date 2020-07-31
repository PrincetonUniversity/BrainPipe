# Analysis scripts for light sheet microscopy for the Brody lab.

Edits made by Emily Jane Dennis (ejdennis@princeton) with significant help from Zahra (zmd@princeton). Forked from PrincetonUniversity/BrainPipe by Tom Pisano, Zahra John D'Uva. It also includes modified scripts from ClearMapCluster and lightsheet_helper_scripts written by the same people. ClearMap Cluster is T. Pisano's parallelization to a cluster of C. Kirst's ClearMap software (https://idisco.info/clearmap/) for use on a cluster using a slurm based scheduler. Written for Python 3.7+. Modifications by Zahra M.  This is using ClearMap 1, but ClearMap2 is now available and we should move to that reasonably soon.

Includes three-dimensional CNN with a U-Net architecture (Gornet et al., 2019; K. Lee, Zung, Li, Jain, & Sebastian Seung, 2017) with added packages developed by Kisuk Lee (Massachusetts Institute of Technology), Nick Turner (Princeton University), James Gornet (Columbia University), and Kannan Umadevi Venkatarju (Cold Spring Harbor Laboratories).

# Emily is doing some heavy maintenance on these scripts, and has not yet updated the README. Her instructions are [here](https://docs.google.com/document/d/1cuNthPY2Z-69SQi9aSwfbgJlHpvQGivhxFtOUmAKOm4/edit?usp=sharing)

### *Dependencies:*
[DataProvider3](https://github.com/torms3/DataProvider3)  
[PyTorchUtils](https://github.com/nicholasturner1/PyTorchUtils)  
[Augmentor](https://github.com/torms3/Augmentor)  
[DataTools](https://github.com/torms3/DataTools)

## *INSTALLATION INSTRUCTIONS*:
* Note that this currently has only been tested on Linux (Ubuntu 16 and 18).
* Things you will need to do beforehand:
	* Elastix needs to be compiled on the cluster - this was challenging for IT here and suspect it will be for your IT as well.

# *Installation Instructions*:
* Things you will need to do beforehand:
	* Elastix needs to be compiled on the cluster - this was challenging for IT here and suspect it will be for your IT as well.
	* This package was made for linux/osx, not windows.

## Create an anaconda python environment
 - use [anaconda](https://www.anaconda.com/download/
- name an [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) 'lightsheet' (in python 3.7+)

```
$ conda create -n lightsheet python=3.7.3
$ pip install cython futures h5py joblib matplotlib natsort numba numpy opencv-python openpyxl pandas scipy scikit-image scikit-learn seaborn SimpleITK tifffile tensorboardX torch torchvision tqdm xlrd xvfbwrapper
```

If on a local machine also do:

```
$ sudo apt-get install elastix
$ sudo apt-get install xvfb
```

If on a local machine, make sure you have all the boost libraries installed (important for working with torms3's DataTools)

```
$ sudo apt-get install libboost-all-dev
```

Navigate to `tools/conv_net` and clone the necessary C++ extension scripts for working with DataProvider3:

```
```

$ git clone https://github.com/torms3/DataTools.git


Go to the dataprovider3, DataTools, and augmentor directories in `tools/conv_net` and run (for each directory):

```
$ python setup.py install
```

## To use TeraStitcher it must be installed locally or on your cluster
* [Download] and unpack(https://github.com/abria/TeraStitcher/wiki/Binary-packages])
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

## Edit: lightsheet/sub_main_tracing.sh file:
* Need to load anacondapy 5.3.1 on cluster (something like):
```
module load anacondapy/5.3.1
```
* Need to load elastix on cluster (something like):
```
module load elastix/4.8
```
* Need to then activate your python environment where everything is installed (if your enviroment is named 'lightsheet' then you do not need to change this):
```
. activate ligthsheet
```
* Check to make sure your slurm job dependecies and match structure is similar to what our cluster uses.

## Edit: lightsheet/slurm_files:
* Each of these needs the same changes as sub_main_tracing.sh file, e.g.
```
module load anacondapy/5.3.1
module load elastix/4.8
. activate ligthsheet
```

* Check/change the resource allocations and email alerts at the top of each .sh file based on cluster and run_tracing.py settings

## Edit: lightsheet/tools/utils/directorydeterminer:
* Add your paths for BOTH the cluster and local machinery

## Edit: lightsheet/tools/conv_net/pytorchutils:
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
	3. these need the same changes as `sub_main_tracing.sh` file, e.g.
```
module load anacondapy/5.3.1
. activate ligthsheet
```

## To run, I suggest:
* Open `run_tracing.py`
* For **each** brain modify:
	* `inputdictionary`
	* `params`
	* **NOTE** we've noticed that elastix (registration software) can have issues if there are spaces in path name. I suggest removing ALL spaces in paths.
* Then, I suggest, using a local machine, run 'step 0' (be sure that `run_tracing.py` is edited is **before**):
```python
preprocessing.generateparamdict(os.getcwd(), **params)
if not os.path.exists(os.path.join(params['outputdirectory'], 'lightsheet')):
	shutil.copytree(os.getcwd(), os.path.join(params['outputdirectory'], 'lightsheet'),
	ignore=shutil.ignore_patterns('^.git'))
```

* **why**: This generates a folder where data will be generated, allowing to run multiple brains on the cluster at once.
* then using the cluster's headnode (in the **new** folder's lightsheet directory generated from the previous step) submit the batch job: `sbatch sub_registration.sh`

# *Descriptions of important files*:

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
		* make sure you have an environment setup under your cluster username named "3dunet" or "lightsheet" that has the dependencies described in the installation instructions
			* NOTE: the environments "3dunet" and "lightsheet" are sometimes used interchangeably in all bash scripts (but represent the same environment)
			* make sure you have the correct environment name in your bash scripts before executing them
		* you will also need CUDA installed under your username; check with IT on how to setup CUDA properly under your cluster username
		* load the modules and environment in the bash script as such:
```
module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate ligthsheet
```

2. else, navigate to tools/conv_net; in the terminal, in the lightsheet environment, run:
```
$ python setup_demo_script.py
$ cd pytorchutils/
$ python demo.py demo models/RSUNet.py samplers/demo_sampler.py augmentors/flip_rotate.py 10 --batch_sz 1 --nobn --noeval --tag demo
```
3. output will be in a 'tools/conv_net/demo/cnn_output' subfolder (as a TIFF)
