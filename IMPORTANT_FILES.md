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