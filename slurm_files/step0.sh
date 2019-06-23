#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 20                 # time (minutes)
#SBATCH -o logs/step0.out        # STDOUT
#SBATCH -e logs/step0.err        # STDERR

module load anacondapy/2.7
module load elastix/4.8
. activate lightsheet

echo "Starting step 0"

xvfb-run python run_tracing.py 0 ${SLURM_ARRAY_TASK_ID} #update dictionary and pickle

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh 

