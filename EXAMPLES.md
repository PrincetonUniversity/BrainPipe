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

# Example: Stitching on a local machine and a cluster
- Download the demo dataset for stitching and unzip it: INSERT URL HERE.
        - `tar -zxvf demo_stitching_dataset_4brainpipe.tar.gz`

This demo dataset contains a 3x3 grid of Z planes from 0-200 taken with two lightsheets (left and right). Each Z plane is spaced by 2 microns and each Z plane has pixels which are 1.63 microns on a side.
- Modify `parameter_dictionary.py` so that `inputdictionary` points to where you downloaded and unzipped the dataset. Make sure to set the following parameters:
        - `"xyz_scale": (1.63,1.63,2)`
        - `"tiling_overlap": 0.1`
        - `"stitchingmethod": "terastitcher"`
        - `"resizefactor": 5`

## Local machine instructions:
Once you have set up your `parameter_dictionary.py` file correctly


