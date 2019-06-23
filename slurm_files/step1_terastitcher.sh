#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 6                      # number of cores
#SBATCH -t 1400                 # time (minutes)
#SBATCH -o logs/step1_terastitcher_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/step1_terastitcher_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 80000


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"


module load anacondapy/2.7
module load elastix/4.8
. activate lightsheet

xvfb-run python run_tracing.py 1 ${SLURM_ARRAY_TASK_ID} #process zplns, check that 1000 > zplns/slurmfactor

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh
#xvfb-run --auto-servernum --server-num=1
