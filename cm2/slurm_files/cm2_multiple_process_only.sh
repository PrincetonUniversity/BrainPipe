#!/bin/env bash
#

#SBATCH -c 1                      # number of cores
#SBATCH -t 20                  # time (minutes)
#SBATCH -o logs/clearmap2_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j.err        # STDERR #add _%a to see each array job

PYTHONPATH="${PYTHONPATH}:/scratch/ejdennis/rat_BrainPipe/ClearMap2/ClearMap"

module load anacondapy/2020.11
. activate cm2

# change these:
declare -a LIST_OF_FOLDERS=("/scratch/ejdennis/cm2_brains/m128/488"
"/scratch/ejdennis/cm2_brains/m128/642"
"/scratch/ejdennis/cm2_brains/X073/488"
"/scratch/ejdennis/cm2_brains/X073/642"
"/scratch/ejdennis/cm2_brains/X077/488"
"/scratch/ejdennis/cm2_brains/X077/642"
"/scratch/ejdennis/cm2_brains/X078/488"
"/scratch/ejdennis/cm2_brains/X078/642")

declare -a LIST_OF_DESTINATIONS=("/scratch/ejdennis/cm2_brains/m128/488"
"/scratch/ejdennis/cm2_brains/m128/642"
"/scratch/ejdennis/cm2_brains/X073/488"
"/scratch/ejdennis/cm2_brains/X073/642"
"/scratch/ejdennis/cm2_brains/X077/488"
"/scratch/ejdennis/cm2_brains/X077/642"
"/scratch/ejdennis/cm2_brains/X078/488"
"/scratch/ejdennis/cm2_brains/X078/642")

module load anacondapy/2020.11
. activate cm2

for (( n=0; n<=${#LIST_OF_FOLDERS[@]}; n++ ))
do
    echo "$n"    
    echo "${LIST_OF_FOLDERS[n]}"
    echo "${LIST_OF_DESTINATIONS[n]}"
    sbatch -p Brody --array=0 slurm_files/cm2_process.sh "${LIST_OF_FOLDERS[n]}" "${LIST_OF_DESTINATIONS[n]}" "smartspim"
done






