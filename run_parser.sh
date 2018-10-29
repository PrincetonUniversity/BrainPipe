#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 12                      # number of cores
#SBATCH -t 300                # time (minutes)
#SBATCH -o parser_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e parser_%a.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 30000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"


module load anacondapy/5.1.0
. activate lightsheet

python run_fwd.py 0 

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh
#xvfb-run --auto-servernum --server-num=1
