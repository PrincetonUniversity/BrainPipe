#!/bin/env bash
#

#SBATCH -c 1                      # number of cores
#SBATCH -t 9                  # time (minutes)
#SBATCH -o logs/clearmap2_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j.err        # STDERR #add _%a to see each array job

PYTHONPATH="${PYTHONPATH}:/scratch/ejdennis/rat_BrainPipe/ClearMap2/ClearMap"

module load anacondapy/2020.11
. activate cm2

# change these:
declare -a LIST_OF_FOLDERS=("/jukebox/LightSheetData/pbibawi/pb_udisco/pb_udisco-E155/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em0_corrected" "/jukebox/LightSheetData/pbibawi/pb_udisco/pb_udisco-E155/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em2_corrected")
declare -a LIST_OF_DESTINATIONS=("/scratch/ejdennis/cm2_brains/E155/488" "/scratch/ejdennis/cm2_brains/E155/642")

module load anacondapy/2020.11
. activate cm2

for (( n=0; n<=${#LIST_OF_FOLDERS[@]}; n++ ))
do
    echo "$n"    
    echo "${LIST_OF_FOLDERS[n]}"
    echo "${LIST_OF_DESTINATIONS[n]}"
    sbatch slurm_files/cm2_all.sh "$LIST_OF_FOLDERS[n]" "$LIST_OF_DESTINATIONS[n]" "smartspim"
done






