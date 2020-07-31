#!/bin/env bash
#
#SBATCH -n 1                      # number of cores
#SBATCH -t 20                 # time (minutes)
#SBATCH -o logs/outmain_registration.out        # STDOUT
#SBATCH -e logs/outmain_registration.err        # STDERR


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

#generate memmap array of full size cell channel data
OUT4=$(sbatch --dependency=afterany:${OUT1##* } slurm_files/cnn_step0.sh "`pwd`")
echo $OUT4

#generate chunks for cnn input
OUT5=$(sbatch --dependency=afterany:${OUT4##* } --array=0-130 slurm_files/cnn_step1.sh "`pwd`")
echo $OUT5

#check if correct number of patches were made
OUT6=$(sbatch --dependency=afterany:${OUT5##* } slurm_files/cnn_step1_check.sh "`pwd`")
echo $OUT6

#run elastix
OUT3=$(sbatch --dependency=afterany:${OUT2##* } --array=0-2 slurm_files/step3.sh)
echo $OUT3


# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully