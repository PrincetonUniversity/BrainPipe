#!/bin/env bash
#
#SBATCH -c 12                 # number of cores
#SBATCH -t 500                 # time (minutes)
#SBATCH -o logs/array_jobs/step4_%a.out        # STDOUT
#SBATCH -e logs/array_jobs/step4_%a.err        # STDERR
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 150000 #150 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"


module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

#TP adding in a random delay between 0 and 30 seconds to help with errors
xvfb-run -w  $(( ( RANDOM % 30 )  + 1 )) python run_clearmap_cluster.py 4 ${SLURM_ARRAY_TASK_ID}

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh
