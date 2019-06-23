#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 4                      # number of cores
#SBATCH -t 600                 # time (minutes)
#SBATCH -o logs/step1_check.out        # STDOUT
#SBATCH -e logs/step1_check.err        # STDERR
#SBATCH --contiguous #used to try and get cpu mem to be contigous

module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

#check to ensure all planes have been processed in step 1
xvfb-run python run_tracing.py 11 ${SLURM_ARRAY_TASK_ID} #process zplns, check that 1000 > zplns/slurmfactor

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh 
#xvfb-run --auto-servernum --server-num=1 
