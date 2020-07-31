#!/bin/env bash
#
#SBATCH -n 12                      # number of cores
#SBATCH -t 500                 # time (minutes)
#SBATCH -o /scratch/zmd/logs/param_sweep_step1_%a.out        # STDOUT
#SBATCH -e /scratch/zmd/logs/param_sweep_step1_%a.err        # STDERR
#SBATCH --mem 150000 #150 gbs

module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

xvfb-run python run_parameter_sweep.py 1 ${SLURM_ARRAY_TASK_ID}

# HOW TO USE:
# sbatch --array=0-20 parameter_sweep.sh
