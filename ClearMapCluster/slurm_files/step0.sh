#!/bin/env bash
#
#SBATCH -n 1                      # number of cores
#SBATCH -t 20                 # time (minutes)
#SBATCH -o logs/step0.out        # STDOUT
#SBATCH -e logs/step0.err        # STDERR

module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

xvfb-run python run_clearmap_cluster.py 0 ${SLURM_ARRAY_TASK_ID} #update dictionary and pickle

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh
