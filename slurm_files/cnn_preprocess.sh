#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o logs/cnn_preprocess_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/cnn_preprocess_%j.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/2020.11
. activate brainpipe

echo "Experiment name:" "`pwd`"

#generate memmap array of full size cell channel data
OUT0=$(sbatch slurm_files/cnn_step0.sh "`pwd`") 
echo $OUT0

#generate chunks for cnn input
OUT1=$(sbatch --dependency=afterany:${OUT0##* } --array=0-130 slurm_files/cnn_step1.sh "`pwd`") 
echo $OUT1

#check if correct number of patches were made
OUT2=$(sbatch --dependency=afterany:${OUT1##* } slurm_files/cnn_step1_check.sh "`pwd`") 
echo $OUT2

#functionality
#go to 3dunet main directory and type sbatch cnn_preprocess.sh [path to lightsheet package output directory]
