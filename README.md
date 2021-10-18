# BrainPipe: Registration and object detection pipelines for three-dimensional whole-brain analysis.

 Includes three-dimensional convolutional neural network (CNN)  with a U-Net architecture (Gornet et al., 2019; K. Lee, Zung, Li, Jain, & Sebastian Seung, 2017) with added packages developed by Kisuk Lee (Massachusetts Institute of Technology), Nick Turner (Princeton University), James Gornet (Columbia University), and Kannan Umadevi Venkatarju (Cold Spring Harbor Laboratories).

### CNN Dependencies (see install instructions below):
[DataProvider3](https://github.com/torms3/DataProvider3)  
[PyTorchUtils](https://github.com/nicholasturner1/PyTorchUtils)  
[Augmentor](https://github.com/torms3/Augmentor)  
[DataTools](https://github.com/torms3/DataTools) 

### Contact: tpisano@princeton.edu, zmd@princeton.edu, ahoag@princeton.edu

# Installation Instructions:
- The pipelines can be run locally or on a slurm-based computing cluster. This package was made for linux/osx, not windows. If running windows we suggest using a virtual machine.
		(1) [Download Virtual Box](https://www.virtualbox.org/wiki/Downloads)
		(2) [Download Linux Ubuntu](https://www.ubuntu.com/download)
		(3) [Install the VM machine](http://www.instructables.com/id/How-to-install-Linux-on-your-Windows/)

## System-wide setup:
If on a local machine:
```
$ sudo apt-get install elastix 
$ sudo apt-get install xvfb 
$ sudo apt-get install libboost-all-dev 
```
* [Download] and unpack terastitcher: https://github.com/abria/TeraStitcher/wiki/Binary-packages and then run:
```
$ bash TeraStitcher-Qt4-standalone-1.10.16-Linux.sh?dl=1
```
Note that the filename may slightly differ from this if the version has changed.
- Modify Path in ~/.bashrc:
```
export PATH="<path/to/software>TeraStitcher-Qt4-standalone-1.16.11-Linux/bin:$PATH"
```
* Check to see if successful
```
$ which terastitcher
```
This should show the path. If it shows nothing, then check the `export` line in your bashrc file to mkae sure it is correct.

If you are planning to run BrainPipe on a computing cluster ask your system administrator to install the dependencies mentioned above on the cluster.

## Python setup
Once you have completed the system-wide requirements, install [anaconda](https://www.anaconda.com/download/) if not already present on your machine or cluster.

The file in this github repository called: `brainpipe_environment.yml` contains the configuration of the specific anaconda environment used for BrainPipe. Install the anaconda environment by running the command:
```
$ conda env create -f brainpipe_environment.yml
```
This will create an anaconda environment called `brainpipe` on your machine. Once, activate the environment by running:
```
$ conda activate brainpipe
```
Navigate to `tools/conv_net` and clone the necessary C++ extension scripts for working with DataProvider3:
```
$ git clone https://github.com/torms3/DataTools.git
```
Go to the dataprovider3 and DataTools directories in `tools/conv_net` and run (for each directory):
```
$ python setup.py install
```
Then go to the augmentor directory in tools/conv_net and run:
```
$ pip install -e .
```

After you have completed the system-wide setup and the Python setup, we suggest going through the [Examples](EXAMPLES.md)


# Example: brain registration on a slurm-based computing cluster
After running through the brain registration on a local machine example, there is not that much more needed to understand how to run brain registration on a cluster. The main difference is that instead of manually running the `main.py` file with python, this file is called from sbatch scripts, which submit the jobs on the cluster. Here are the steps for setting this up.

- The file `parameter_dictionary.py` still needs to be set up in the same way as for the local machine. 
- The file `registration_pipeline.sh` is a bash script that runs an sbatch script for each step in the pipeline. These sbatch jobs are dependent on each other so that each subsequent step only runs if the previous finished without errors. Each sbatch script lives in `slurm_files/`. 
- Edit the `step0.sh`, `step1.sh`, `step2.sh` and `step3.sh` files so that the lines:
```
module load anacondapy/2020.11
module load elastix/4.8
``` 
reflect the module names and versions that your cluster administrator installed for you to enable you to create your conda `brainpipe` environment and run elastix on the cluster. 
- edit `registration_pipeline.sh` so that the number of array jobs in step1 are sufficient to cover all z planes. Remember that each array job will process `slurmjobfactor` z planes (see your `parameter_dictionary.py`, default is 50).
- edit `registration_pipeline.sh` so that the number of array jobs in step2 corresponds to the number of channels you want to downsize and register.
- edit `registration_pipeline.sh` so that the array jobs in step3 correspond to the types of registration you want to perform (see [step3](#step3))

Edit: lightsheet/sub_main_tracing.sh file:
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
. activate <<<your python environment>>>
```
* Check to make sure your slurm job dependecies and match structure is similar to what our cluster uses.
 
## Edit: lightsheet/slurm_files:
* Each of these needs the same changes as sub_main_tracing.sh file, e.g.
```
module load anacondapy/5.3.1
module load elastix/4.8
. activate <<<your python environment>>>
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
. activate <<<your python environment>>>
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
	* `.py` file to be used to manage the parallelization _of CNN preprocessing and postprocessing_ to a SLURM cluster
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
. activate <<<your python environment>>>
```

2. else, navigate to tools/conv_net; in the terminal, in the lightsheet environment, run:
```
$ python setup_demo_script.py
$ cd pytorchutils/
$ python demo.py demo models/RSUNet.py samplers/demo_sampler.py augmentors/flip_rotate.py 10 --batch_sz 1 --nobn --noeval --tag demo
```
3. output will be in a 'tools/conv_net/demo/cnn_output' subfolder (as a TIFF)

# CNN paralellization

* for whole brain, cellular resolution image volumes (> 100 GB), the neural network inference is parallelized across multiple chunks of image volumes, and stitched together by taking the maxima at the overlaps of the chunks after inference.
* the chunks are made by running `cnn_preprocess.sh` on a CPU based cluster
* chunks can then be run for inference on a GPU based cluster (after transfer to the GPU based cluster server or on a server that has both CPU and GPU capabilities)
	* by modifying paths in `tools/conv_net/pytorchutils/run_chnk_fwd.py`
	* by then navigating to `tools/conv_net/pytorchutils` and submitting an array batch job
		* the range ("0-150") will depend on how many chunks were made for the whole brain volume, which are typically 80-200
```
sbatch --array=0-150 slurm_scripts/run_chnk_fwd.sh
````
