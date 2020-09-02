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
$ sudo apt-get install xvfb
$ sudo apt-get install libboost-all-dev
```

To install elastix, follow the instructions in the manual under the easy way, not the "super easy" way

if you use the 'easy way' but have a modern computer, your gcc version may be too high. For this, you'll need at least ITK 5.0 which means you need to use elastix version 5, not 4.8. The following worked on Ubuntu 18 with two GeForce RTX 2070 SUPERs.

- made two dirs: ITK-build and elastix-build
- added ITKSNap 5
  file:///tmp/mozilla_emilyjanedennis0/InsightSoftwareGuide-Book1-5.1.0.pdf
  extracted downloaded .tar.gz and .tar.bz2 files to those directories
  in ITK-build, typed  `cmake ITK-build`
  then `sudo make install`
  in elastix-build, `cmake elastix-build` failed, so I went into the folder and found the CMakeFiles.txt
  `cd elastixfolder
  nano CMakeLists.txt`
  and added `list( APPEND CMAKE_PREFIX_PATH "/home/emilyjanedennis/Desktop/Apps/ITK-build/" )`

  note: I had to remove line 76 from `elastix-5.0.0/Components/Resamplers/MyStandardResampler/elxMyStandardResampler` which referred to an  undefined type called PointType which was throwing an exception during the make install process for elastix

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
If there are errors in these steps, usually it's
    1. regex needs to be edited
    2. elastix isn't installed properly (try `which elastix`) or is missing from bashrc
    3. terastitcher isn't installed properly (try `which terastitcher`) or is missing from bashrc
**things to do before running**
- edit run_tracing.py to poit to the appropriate directories and use the correct parameters for your imaging. especially pay attention to:
- systemdirectory
  - if you haven't edited directorydeterminer() appropriately, nothing will work
- inputdictionary
  - point to the raw images file, they should be in the format like
    `10-50-16_UltraII_raw_RawDataStack[00 x 00]_C00_xyz-Table Z0606_UltraII Filter0000.ome.tif`
  - if the format of your images differs, you'll need to edit the regex expression called in run_tracing (find it in tools/imageprocessing/preprocessing under `def regex_determiner`)
  - if you have more than one channel imaged in this folder, add channels. Examples:
    - one channel in the folder: `[["regch", "00"]]`
    - two chhannels in the folder `[["regch", "00"], ["cellch","01"]]`
- under params edit:
  - outputdirectory
        - give the path where you want things saved. I usually make it the animal's name (e.g. E001) and store temporarily in scratch `scratch/ejdennis/e001`   or more permanently ` /LightSheetData/brodyatlas/processed/e001`
  - xyz_scale
    - usually either (5,5,10) for data collected with the 1.3x objective or (1.63,1.63,10) if collected with the 4X objective
  - stitchingmethod
    - I keep this as "terastitcher"
  - AtlasFile
    - point to the file you want to register the brain to - usually this will be either the Waxholm MRI atlas `/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_T2star_v1.01_atlas.tif` or our atlas
  - annotationfile
    - the annotation file that describes the above brain, e.g. `/jukebox/LightSheetData/brodyatlas/atlas/for_registration_to_lightsheet/WHS_SD_rat_atlas_v3_annotation.tif`
  - resizefactor
    - usually 5 for 4x images, 3 for 1.3x images
  - slurmjobfactor
    - we keep this at 50, it's the number of files processed in step1 of run_tracing and slurm_files/step1.sh
  - transfertype
    - MAKE SURE THIS IS "copy" or else your data may get deleted, there are backups, but it's really annoying. Avoid at all costs, you can always clean up and delete things later.
  - you'll also want to check that in the __main__ run the systemdirectory points to your use case. For example, my scripts see if I'm running locally ("/home/emilyjanedennis/")

**to run**
if on spock, *after editing run_tracing.py*, go to your rat_BrainPipe folder, and run
    `sbatch slurm_scripts/step0.sh`
then go to your outputdirectory (e.g. /scratch/ejdennis/e001) that you specified in run_tracing.py
    `cd /scratch/ejdennis/e001`
there should now be a directory called lightsheet, go there
    `cd lightsheet`
run the pipeline from this folder (this is useful because it allows you to keep track of the exact parameters you used to run, and also parallelize by launching jobs for different brains at once on the cluster)
    `sbatch sub_registration.sh`

That's all! If everything runs successfully, you'll end up with a param_dict.p, LogFile.txt, two new resized tiffstacks, a mostly empty folder called injections, a folder called full_sizedatafld with the individual stitched z planes for each channel, and an elastix folder with the brains warped to the atlas defined in run_tracing.py AtlasFile

### 2. Make an atlas
If you have a group of stitched brains (e.g. e001/full_sizedatafld, e002/full_sizedatafld, and e003/full_sizedatafld), you can make an average brain for an atlas. Our rat atlas is for our lab, and therefore is made of only male (defined operationally by external genitalia) brains. However, we wanted to test our results and publish including female (similarly operationally defined) brains. Therefore we perfused, cleared, and imaged three female brains and created an atlas.

To make your own atlas, use the  `rat_atlas` folder.
  1. Edit `mk_ls_atl.sh` amd `cmpl_atl.sh` to use your preferred slurm SBATCH headings
  2. Edit `step1_make_atlas_from_lightsheet_images`
    - edit sys.path.append in the import section to reference your rat_BrainPipe cloned git repo
    - main section variables:
      - src should be where your folders (typically named by animal name, e.g. e001) live, these folders should each have full_sizedatafld folders in them
      - dst - where you want to save things. If you have a nested structure, make sure the parent structure exists (e.g.if you want to save in /jukebox/scratch/ejdennis/female_atlas/volumes, make sure /jukebox/scratch/ejdennis/female_atlas already exists)
      - brains should be the list of names of the brains you wish to use, corresponding to the names of the folders in dst that you want to average
  3. Run `sbatch --array=0-2 mk_ls_atl.sh` for three brains, --array=0-9 for 10 brains, etc.
  4. Edit `step2_compie_atlas.py`
    - edit sys.path.append in the import section to reference your rat_BrainPipe cloned git repo
    - edit main section variables:
      - src - should be the same as in step2
      - brains - should be the same as in step2
      - output_fld - should be a *different* folder than in step2: I like to place them in the same parent folder
      - parameterfld - this should point to a folder containing the affine/bspline transforms you want to use
  5. Run `sbatch --array=0-2 cmpl_atl.sh` for three brains, --array=0-9 for 10 brains, etc.
  6. Edit `step3_make_median.py`
    - edit the sys.path.append in the import section
    - in main, edit variables to match step2_compile_atlas
  7. Either locally or on the cluster head node (module load anacondapy/5.3.1), use export SLURM_ARRAY_TASK_ID=0, activate the lightsheet conda environment, and run `step3_make_median.py`


### 3. Put a brain in atlas space
if you have already "Made a stitched whole-brain", you may already have your brain in atlas space, depending on what you specified as the AtlasFile. If you have a tiff stack and you want to register it to an atlas file, you can use `elastix_to_pra.py`
    - change the mv to be your "moving image" (the brain tiffstack) and fx to your "fixed image" (the atlas volume)
    - change the output directory to where you want your elastix files and newly aligned tiff saved
   - change the outputfilename - this will be a resized mv file that is 140% the size of fx and is what is actually used for the alignment

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
