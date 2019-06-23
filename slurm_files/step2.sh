#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 10                      # number of cores
#SBATCH -t 360                 # time (minutes)
#SBATCH -o logs/step2_%a.out        # STDOUT
#SBATCH -e logs/step2_%a.err        # STDERR
#SBATCH --contiguous


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

xvfb-run python run_tracing.py 2 ${SLURM_ARRAY_TASK_ID} #combine stacks into single tifffiles

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh 

