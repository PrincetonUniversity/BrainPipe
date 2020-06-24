#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 8                      # number of cores
#SBATCH -t 200                # time (minutes)
#SBATCH -o logs/array_jobs/cnn_step1_chnk%a_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/array_jobs/cnn_step1_chnk%a_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 30000 #30 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.3.1
. activate lightsheet

echo "Experiment name:" "`pwd`"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

python cell_detect.py 1 ${SLURM_ARRAY_TASK_ID} "`pwd`" 
