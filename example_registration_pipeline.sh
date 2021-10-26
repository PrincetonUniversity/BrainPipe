#!/bin/env bash


#set up dictionary and save
OUT0=$(sbatch slurm_files/step0.sh) 
echo $OUT0

#process zplns
OUT1=$(sbatch --dependency=afterany:${OUT0##* } --array=0-14 slurm_files/step1.sh) 
echo $OUT1

#combine stacks into single downsized tifffiles
OUT2=$(sbatch --dependency=afterany:${OUT1##* } --array=0-1 slurm_files/step2.sh) 
echo $OUT2

#run elastix
OUT3=$(sbatch --dependency=afterany:${OUT2##* } --array=0-1 slurm_files/step3.sh) 
echo $OUT3


# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully
