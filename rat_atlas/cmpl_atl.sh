#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 12                      # number of cores
#SBATCH -t 1000                # time (minutes)
#SBATCH -o /scratch/zmd/logs/cmpl_atl.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/cmpl_atl.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 80000 #80 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

echo "Array Index: $SLURM_ARRAY_TASK_ID"

python step2_compile_atlas.py ${SLURM_ARRAY_TASK_ID}
