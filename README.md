# Analysis scripts for light sheet microscopy for the Brody lab.

Edits made by Emily Jane Dennis (ejdennis@princeton) with significant help from Zahra (zmd@princeton). Forked from PrincetonUniversity/BrainPipe by Tom Pisano, Zahra Dhanerawala, John D'Uva. It also includes modified scripts from ClearMapCluster and lightsheet_helper_scripts written by the same people. ClearMap Cluster is T. Pisano's parallelization to a cluster of C. Kirst's ClearMap software (https://idisco.info/clearmap/) for use on a cluster using a slurm based scheduler. Written for Python 3.7+. Modifications by Zahra.  

This is using ClearMap 1, but ClearMap2 is now available and we should move to that reasonably soon.

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
If on the cluster, and typing which terastitcher can't find terastitcher, try adding the following to your path

```
export PATH="/usr/people/pnilsadmin/TeraStitcher-Qt4=standalone-1.10.11-Linux/bin$PATH"
```

If on a local Ubuntu machine also install elastix, xvfb, Terastitcher, and make sure you have all boost libraries installed for DataTools:

```
$ sudo apt-get install elastix
$ sudo apt-get install xvfb
$ sudo apt-get install libboost-all-dev
```

to properly use elastix you need a few more steps - specifically you need the OpenCL on a local Ubuntu 18 machine with 2 NVIDIA GeForce RTX 2070 SUPER GPUs in August 2020, these steps worked:

    # clinfo is opencl
    sudo apt install -y clinfo
    # the next steps check and update nvidia graphics drivers
    sudo add-apt-repository ppa:graphics-drivers/ppa
    apt search nvidia-driver
    # below, I chose 450 because that was the most recent driver, ID'd from the step above
    sudo apt install nvidia-driver-450

or follow the instructions in the manual under the easy way, not the "super easy" way

final note on elastix install: if you use the 'easy way' but have a modern computer, your gcc version may be too high. For this, you'll need at least ITK 5.0

[Download](https://github.com/abria/TeraStitcher/wiki/Binary-packages) TeraStitcher-installer. Move file to wherever you want Terastitcher to live, cd into that directory, and then:
```
$ bash TeraStitcher-Qt4-standalone-1.10.18-Linux
```
* Modify Path in ~/.bashrc:

```
export PATH="<path/to/software>/TeraStitcher-Qt4-standalone-1.10.18-Linux/bin:$PATH"
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

### If not at Princeton - make sure your cluster works similarly
* If not, change .sh in main folder, /slurm_scripts in the main rat_BrainPipe, and ClearMapCluster folders like sub_main_tracing.sh file, e.g.
```
module load anacondapy/5.3.1
module load elastix/4.8
. activate ligthsheet
```

* Check/change the resource allocations and email alerts at the top of each .sh file based on cluster and run_tracing.py settings

### Edit: lightsheet/tools/utils/directorydeterminer:
* Add your paths for BOTH the cluster and local machinery

## To train a CNN to identify features (like somas)

0. Make test sets
    Emily made a document with explicit instructions and examples
    which can be found [here](https://docs.google.com/document/d/1f5owGhyJiL2dNqIMZ_zwM96QTBm2_WKqoY5HwBRAMVA/edit)

1. Find parameters to use
2. Train network
3. Validate and Benchmark Network
    - Detailed instructions on steps 1-3 can be found [here](https://docs.google.com/document/d/1cuNthPY2Z-69SQi9aSwfbgJlHpvQGivhxFtOUmAKOm4/edit#)

4. Run inference on whole brains
**MAKE SURE YOU HAVE CHECKED that the model params are the same in run_chnk_fwd as they were when you trained the network!**

- on spock, make input and patched arrays
    - in `run_tracing.py` edit the inputdirectory, outputdirectory, and check the atlas and annotation files are correct.
        - MAKE SURE you have already changed `directorydeterminer` in `tools/utils` and run step0
        - check the header for `sub_main_tracing_cnn.sh` and make sure for step1, the array job number matches your zplanes. There are 50 jobs/array, so --array=0-12 means 13 sets of 50 (650) zplanes will be run.
        - submit job to cluster using `sbatch --array=0 sub_main_tracing_cnn.sh`
        - when finished, use [Globus](https://www.globus.org/) to transfer files to tigress. You only need to transfer the:
            - cnn_param_dict.csv
            - input_chnks folder
            - lightsheet folder
            - LogFile.txt
            - output_chnks (an empty folder that will be filled on tigress)
            - param_dict.p
            - **NOTE** you actually probably don't need all of these and only need input_chnks and output_chnks but the others are small and used a bunch of places.
        - on tigress, go into ratname/lightsheet and check that `tools/conv_net/pytorchutils/run_chunked_fwd.py` has the correct paths for tigress. Also check that `tools/conv_net/pytorchutils/slurm_scripts/run_chnk_fwd.sh` uses the correct model/checkpoint
        - cd into `tools/conv_net/` and run with `sbatch --array=0 slurm_scripts/run_chnk_fwd.sh`
        - when finished, use [Globus](https://www.globus.org/) to transfer back to spock or `scp -r <username>@tigergpu.princeton.edu:/<folder-location> <username>@spock.princeton.edu:/<folder-location>`
        - run `cnn_postprocess.sh` which reconstructs and uses connected components to find cell measures

5. Visualize network outputs - *documentation in progress*

6. To compare to ClearMap
    See next section, has steps on how to run ClearMap 1, visualize, and compare ouputs.

**NOTE** there is a demo for the CNN to test if your paths are correct and everything is installed properly, see end of document

## To use ClearMap to identify cell centers
**NOTE** there is a new [ClearMap!](https://github.com/ChristophKirst/ClearMap2) These instructions use the old ClearMap 1
0. Make test sets
- If you have already made test sets to train a CNN, use those.

1. Find parameters
- See `ClearMapCluster/ parameter_sweep.ipynb`

2. Run ClearMap on whole brains *documentation in progress*

3. Visualize outputs *documentation in progress*

4. Compare to CNN (if desired) *documentation in progress*
    - on training data
    - on whole brains

## Using raw lightsheet images to:

### 1. Make a stitched, whole-brain
*documentation in progress*

### 2. Make an atlas
- for example,

### 3. Put a brain in atlas space
*documentation in progress*

#### Use case 1: adding published atlas information to our atlas space
One use of this can be to take an existing atlas that has parcellations of the brain and put it into our atlas space. This type of 'layer' can be visualized in Neuroglancer (see below) and also can be used to, for example, ID cell centers in different brain regions after tracing.

To do this
- get your atlas into sagittal, tiff format
- Run step3 -

#### Use case 2: putting a single experimental brain into our atlas space

#### other todos  
 - visualize warping procedure
 - identify locations on images and transform to bregma coordinates

### 4. Add annotations to an atlas
*documentation in progress*

## Using Neuroglancer to visualize outputs
- [python notebook from Austin](https://github.com/PrincetonUniversity/lightsheet_helper_scripts/blob/master/neuroglancer/brodylab_MRI_atlas_customizations.ipynb)


# File structure and descriptions

## rat_BrainPipe

These files are the main files used for lightsheet manipulations required for all volumes collected- things like stitching lightsheets and tiles.
- cell_detect.py
* `cell_detect.py`:
	* `.py` file to be used to manage the parallelization _of CNN preprocessing_ to a SLURM cluster
	* params need to be changed per cohort.
	* see the tutorial for more info.
- run_tracing.py
- sub_main_tracing.sh
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
		* `cnn_preprocess.sh`
- sub_registration.sh
- sub_registration_terastitcher.sh
- sub_update_registration.sh

> slurm_files

- in all, check the headers, and make sure you have the correct array numbers, partition, etc.

> parameter folder

- This contains the four parameter files for elastix warping: one affine and three bspline. This is different for mice (which only only uses two: one affline and one bspline)

> tools
-  \_\_init\_\_.py we use for initializing
- analysis: These are mostly for mouse work but could be useful in the future as guides for things to do, so keeping them
- expression_mask: Additional useful plotting tools, though made for mice, keeping for now
- imageprocessing: Useful plotting and processing tools for individual volumes, keeping for now
- objectdetection
    - detection_qc.py is useful for plotting cell centers onto volumes for quality control and nice plots
    - find_injection.py is useful for finding contours of a probe/fiber covered in CM DiI
    - inj_detect.py Similar to find_injection, probably don’t need both, uses connected components analysis
- registration
    - steps to register volumes, transform coordinates
- utils: lots of scripts live here that are used lots of places!

* tools: convert 3D STP stack to 2D representation based on colouring
  * imageprocessing:
	* `preprocessing.py`: functions use to preprocess, stitch, 2d cell detect, and save light sheet images
  * analysis:
	* `allen_structure_json_to_pandas.py`: simple function used to generate atlas list of structures in coordinate space
	* other functions useful when comparing multiple brains that have been processed using the pipeline

>> **conv_net**
this houses all the important CNN files!
    - augmentor
    - dataprovider3
    - DataTools
    - **pytorchutils**
        - slurm_scripts
        - balance.py
        - demo.py
        - forward.py
        - layers.py
        - losee.py
        - run_chnk_fwd.py
        - run_exp.py
        - run_fwd.py
        - train.py


> logs

- file

> ## ClearMapCluster
- clearmap_sweep_for_comp
- parameter_sweep.ipynb
- README.md
- run_clearmap_cluster.py
- run_parameter_sweep.py
- spot_detection_on_annotated_volumes
- sub_clearmap_cluster.sh

>> ClearMap
- copy of ClearMap with some edits as outlined in the ClearMapCluster README.
- parameter_file.py
    - this you will need to change

>> docs

>> logs
- this is where your slurm output logs will go. It also has a sub-folder called array_jobs where slurm array job err/out messages will live.

>> **slurmfiles**
this is where most of the slurm files live.
-

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
