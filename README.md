### Analysis scripts by tpisano@princeton.edu for light sheet microscopy and the cerebellar tracing project using a slurm based computing cluster. 

# *Installation Instructions*:
* Things you will need to do beforehand:
	* Elastix needs to be compiled on the cluster - this was challenging for IT here and suspect it will be for your IT as well.
	* After downloading this package onto your data server (where the cluster has access to it), you will need to install the following depencies. I suggest using an python environment to do this.
	* This package was made for linux/osx, not windows. If running windows I would suggest using a virutal machine.
		(1) [Download Virtual Box](https://www.virtualbox.org/wiki/Downloads)
		(2) [Download Linux Ubuntu](https://www.ubuntu.com/download)
		(3) [Install the VM machine](http://www.instructables.com/id/How-to-install-Linux-on-your-Windows/)
 
## Create an anaconda python environment (Install [anaconda](https://www.anaconda.com/download/) if not already):
### I suggest naming the [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) 'lightsheet' to help with setup.
* `pip install SimpleITK futures xvfbwrapper xlrd openpyxl`
* `conda install joblib scipy scikit-image scikit-learn seaborn tqdm psutil numba natsort`
* `conda install -c menpo opencv` (opencv 3+; if this fails then try: `conda install -c conda-forge opencv`)
* `sudo apt-get install elastix` (if on local machine)
* `sudo apt-get install xvfb` (if on local machine)

## To use TeraStitcher it must be installed locally or on your cluster
* Download and unpack:
	* [Download here](https://github.com/abria/TeraStitcher/wiki/Binary-packages])
	* `$ bash TeraStitcher-Qt4-standalone-1.10.16-Linux.sh?dl=1`
* Modify Path in ~/.bashrc:
	* `export PATH="<path/to/software>TeraStitcher-Qt4-standalone-1.10.11-Linux/bin:$PATH"`
* Check to see if successful
	* open new terminal window
	* `$ which terastitcher`


## Edit: lightsheet/sub_main_tracing.sh file:
* Need to load anacondapy 5.3.1 on cluster (something like):
	* `module load anacondapy/5.3.1`
* Need to load elastix on cluster (something like):
	* `module load elastix/4.8`
* Need to then activate your python environment where everything is installed (something like):
	* `. activate <<<your python environment>>>`
		* if your enviroment is named 'lightsheet' then you do not need to change this.
* Check to make sure your slurm job dependecies and match structure is similar to what our cluster uses.
 
## Edit: lightsheet/slurm_files:
* Each of these needs the same changes as sub_main_tracing.sh file: e.g.:
 
	* `module load anacondapy/5.3.1`
	* `module load elastix/4.8`
	* `. activate <<<your python environment>>>`
		* if your enviroment is named 'lightsheet' then you do not need to change this.
* Check/change the resource allocations and email alerts at the top of each .sh file based on cluster and run_tracing.py settings
 
## Edit: lightsheet/tools/utils/directorydeterminer:
* Add your paths for BOTH the cluster and local machinery
 
## (optional) Edit: lightsheet/tools/utils/io: 
* Find the function load_kwargs
* Replace system_directories = ['/jukebox/', '/mnt/bucket/labs/', '/home/wanglab/'] with your possible system directories

### To run, I suggest:
* Open run_tracing.py
* For **each** brain modify:
	* inputdictionary
	* params
	* **NOTE** we've noticed that elastix (registration software) can have issues if there are spaces in path name. I suggest removing ALL spaces in paths.
* Then, I suggest, using a local machine, run 'step 0' (be sure that run_tracing.py is **before**):
	* `preprocessing.generateparamdict(os.getcwd(), **params)` 
	* `if not os.path.exists(os.path.join(params['outputdirectory'], 'lightsheet')): shutil.copytree(os.getcwd(), os.path.join(params['outputdirectory'], 'lightsheet'), ignore=shutil.ignore_patterns('^.git'))`
	* **why**: This generates a folder where data will be generated, allowing to run multiple brains on the cluster at once.
* then using the cluster's headnode (in the **new** folder's lightsheet directory generated from the previous step) submit the batch job: sbatch sub_main_tracing.sh


# *Descriptions of important files*:

* *sub_registration.sh:* or *sub_registration_terastitcher.sh*
	* .sh file to be used to submit to a slurm scheduler
	* this can change depending on scheduler+cluster but generally batch structure requires 2 variables to pass to run_tracing.py:
		* stepid = controlling which 'step' to run
		* jobid = controlling which the jobid (iteration) of each step
	* Steps:
		* 0: set up dictionary and save; requires a single job (jobid=0)
		* 1: process (stitch, resize, 2D cell detect, etc) zplns (using tpisano lightsheet package), ensure that 1000 > zplns/slurmfactor. Typically submit 1000 jobs (jobid=0-1000)
		* 2: resample and combine; typically submit 3 jobs (requires 1 job/channel; jobid=0-3)
		* 3: registration via elastix

* *run_tracing.sh:*
	* .py file to be used to manage the parallelization to a SLURM cluster
	* inputdictionary and params need to be changed for each brain
	* the function lightsheet.tools.utils.directorydeterminer.directorydeterminer *REQUIRES MODIFICATION* for both your local machine and cluster. This function handles different paths to the same file server.
	* generally the process is using a local machine, run step 0 (be sure that files are saved *BEFORE( running this step) to generate a folder where data will be stored
	* then using the cluster's headnode (in the new folder's lightsheet directory generated from the previous step) submit the batch job: sbatch sub_registration.sh

* tools: convert 3D STP stack to 2D representation based on colouring
  * imageprocessing: 
	* preprocessing.py: functions use to preprocess, stitch, 2d cell detect, and save light sheet images
  * analysis:
	* allen_structure_json_to_pandas.py: simple function used to generate atlas list of structures in coordinate space
	* other functions useful when comparing multiple brains that have been processed using the pipeline

* supp_files:
  * gridlines.tif, image used to generate registration visualization
  * allen_id_table.xlsx, list of structures from Allen Brain Atlas used to determine anatomical correspondence of xyz location.

* parameterfolder:
  * folder consisting of elastix parameter files with prefixes "Order<#>_" to specify application order


