#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 10                     # number of cores
#SBATCH -t 100                # time (minutes)
#SBATCH -o logs/array_jobs/cnn_step3_job%a_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/array_jobs/cnn_step3_job%a_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 100000 #100 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/2020.11
. activate brainpipe

echo "Experiment name:" "`pwd`"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

python cell_detect.py 3 ${SLURM_ARRAY_TASK_ID} "`pwd`" 
