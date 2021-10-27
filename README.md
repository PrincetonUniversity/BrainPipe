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
First, clone this repository.

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

Edit tools/utils/directorydeterminer:
* Add your paths for BOTH the cluster and local machinery