#!/bin/env bash
#
#SBATCH -n 1                      # number of cores
#SBATCH -t 20                 # time (minutes)
#SBATCH -o logs/update_registration.out        # STDOUT
#SBATCH -e logs/update_registration.err        # STDERR


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"


module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

#set up dictionary and save
OUT0=$(sbatch --array=0 slurm_files/step0.sh)
echo $OUT0

#run elastix, note this will not update any orientation changes, useful for new atlas
OUT3=$(sbatch --dependency=afterany:${OUT0##* } --array=0-2 slurm_files/step3.sh)
echo $OUT3


# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully
