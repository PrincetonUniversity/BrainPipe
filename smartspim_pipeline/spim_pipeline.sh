#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o /scratch/zmd/logs/spim_pipeline_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/spim_pipeline_%j.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/5.3.1
. activate lightsheet

echo "Experiment name:" "$1"
echo "Storage directory:" "$2"

#import
OUT0=$(sbatch ts_smartspim_import.sh "$1")
echo $OUT0

#displacement computation
OUT1=$(sbatch --dependency=afterok:${OUT0##* } ts_smartspim_compute.sh "$1")
echo $OUT1

#thresholding
OUT2=$(sbatch --dependency=afterok:${OUT1##* } ts_smartspim_proj.sh "$1")
echo $OUT2

#merge
#make stitched folder
mkdir $2
OUT3=$(sbatch --dependency=afterok:${OUT2##* } ts_smartspim_merge.sh "$1" "$2")
echo $OUT3

#downsize stitched images
OUT4=$(sbatch --dependency=afterok:${OUT3##* } spim_downsize.sh "$2")
echo $OUT4

#do inverse registration
OUT5=$(sbatch --dependency=afterok:${OUT4##* } spim_register.sh "$1")
echo $OUT5

#functionality
#go to smartspim pipeline folder and type smartspim_stitch.sh [path to terstitcher folder hierarchy] [destination of stitc$



