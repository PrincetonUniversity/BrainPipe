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
<<<<<<< HEAD
=======
# Example: brain registration on a local machine 
---
This example is useful to run through once even if you ultimately plan to run the pipeline on a computing cluster.
- The file `parameter_dictionary.py` sets up a dictionary containing all of the information that will be needed to run the code. Edit this file according to its documentation.
- The file `main.py` actually runs the registration pipeline and imports the file `parameter_dictionary.py`. The registration pipeline has four steps, where the step index is the first command line argument you pass to `main.py`. The first step, "step 0", in the pipeline is a bookkeeping step. Once you have edited `parameter_dictionary.py` run this step like:

### Step 0:
```python
python main.py 0
# or: python main.py 0 2>&1 | tee logs/step0.log # if you want to log output to a file
```
This will create the `outputdirectory` directory that you set in `parameter_dictionary.py` and write a few files and sub-directories in there. 
### Step 1:
```python
python main.py 1 $jobid
```
If any channels of your raw data consist of multiple light-sheets, this step will blend them into a single sheet for each Z plane. Also, if you set `"stitchingmethod": "terastitcher"` in `parameter_dictionary.py`, this step will stitch together multiple tiles into a single brain volume. Note that there is a second command line argument passed to `main.py` for this step, $jobid. This parameter references the slurm array job id index on a slurm-based computing cluster, but it used if running the pipeline on a local computer as well. If you need to perform stitching this `jobid` needs to be provided for each channel specified in your `parameter_dictionary.py` file. For example, if you have two channels then you would run:
```python
python main.py 1 0 # blends and stitches first channel
# later, run:
python main.py 1 1 # blends and stitches second channel
```

If not performing stitching, then the `jobid` parameter is used to index which chunk of Z planes to work on. The `slurmjobfactor` parameter in `parameter_dictionary.py` determines how many planes to work on for each jobid, so if `slurmjobfactor: 50` (the default), then for `jobid=0` Z planes 0-49 are processed. For `jobid=1` Z planes 50-99 are processed, and so on. You will need to run this step with multiple jobids until all of your Z planes are processed. Note again that this is only needed if you are not stitching. In that case, the blending happens internally before stitching.

Even if you do not need to stitch or blend your data, you still need to run this step, as it creates files that later steps in the pipeline read. The (optionally) stitched and blended Z planes created during this step will be created in a sub-directory called `full_sizedatafld/` inside your `outputdirectory`. Each channel will have its own sub-directory inside of this directory. These Z planes will have the same voxel resolution as your raw data.

### Step 2:
```python
python main.py 2 $jobid
```
For each channel you specified in `parameter_dictionary.py`, this step will downsize the volume to dimensions that are closer to the dimensions of the reference atlas volume you set in `parameter_dictionary.py`, in preparation for the registration in the following step. It will also reorient your data, if necessary, to the same orientation as the reference atlas volume. Here the `jobid` is used to index which channel to downsize, so run this step for each channel index like:

```python
python main.py 2 0
python main.py 2 1
```
If you have two channels in your `parameter_dictionary.py`.

For each channel, a `*_resized_chXX.tif` will be created in the `outputdirectory`. These files are oriented the same as the reference atlas and downsized in your original data x and y dimensions by a factor of `resizefactor` that you set in `parameter_dictionary.py`. 


### Step 3:
```python
python main.py 3 $jobid
```
This step first resamples the downsized files from Step 2 so that they are 1.4x the size of reference atlas in x, y and z dimensions. These resampled and downsized files are saved as `*_resized_resampledforelastix_chXX.tif` in `outputfolder` during this step. These files are then used as the inputs for the registration to the reference atlas specified in `parameter_dictionary.py`. Whichever channel you set as the `regch` in `parameter_dictionary` will be directly registered to the atlas, since this is often the autofluorescence channel which is closest in appeareance to the atlas. The other channels, if set as `injch` or `cellch` in `parameter_dictionary.py` will first be registered to the `regch` channel and then registered to the atlas. This two-step registration process for non-autofluorescent channels typically results in a better final registration of these channels to the reference atlas. 

In this step, the `jobid` command line argument references the channel type via: 
```
jobid = 
        0: 'normal registration'
        1: 'cellchannel'
        2: 'injchannel`
```
Therefore if you run:
```python
python main.py 3 0
```

A directory called `elastix` will be created in your `outputdirectory`, which will contain the registration results. The `result.0.tif` and `result.1.tif` in this directory refer to registration channel volume that has been registered to the reference atlas space. `result.0.tif` is the volume after an affine transformation, `result.1.tif` is the volume after affine + bspline transformation, and is usually the more accurate result. If there are non-registration channels in your `parameter_dictionary.py` file, it will create sub-directories for each of these channels, e.g. `elastix/_resized_ch01/sig_to_reg`. These directories will contain the registration results between the non-registration channel and the registration channel to be used for the two-step registration process between non-registration channel and the atlas. 

Using `jobid=0` will allow you to register the brain volumes to the atlas, but often it is of interest to register cells or other detected objects in a non-registration image channel to the atlas. That is what the other `jobid` values are for. For example:
```python
python main.py 3 1
```
will create a folder called `elastix_inverse_transform` in your `outputdirectory` containing the inverse transforms of the normal registration achieved with `jobid=0`. These inverse transforms are necessary for transforming coordinates in an image to the atlas coordinate space. 


After you have completed the system-wide setup and the Python setup, we suggest going through the [Examples](EXAMPLES.md)
