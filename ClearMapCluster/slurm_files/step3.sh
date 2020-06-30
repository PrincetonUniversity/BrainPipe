#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 14                 # number of cores
#SBATCH -t 700                 # time (minutes)
#SBATCH -o logs/step3_%a.out        # STDOUT
#SBATCH -e logs/step3_%a.err        # STDERR
#SBATCH --contiguous #used to try and get cpu mem to be contigous


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"



module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

xvfb-run python run_clearmap_cluster.py 3 ${SLURM_ARRAY_TASK_ID} 

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh 

