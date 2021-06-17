#!/bin/env bash
#

#SBATCH -c 1                      # number of cores
#SBATCH -t 20                  # time (minutes)
#SBATCH -o logs/clearmap2_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j.err        # STDERR #add _%a to see each array job


module load anacondapy/2020.11
. activate cm2

# change these:
declare -a LIST_OF_FOLDERS=("/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-005/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected/")

declare -a LIST_OF_DESTINATIONS=("/scratch/ejdennis/cm2_brains/h234/488")

module load anacondapy/2020.11
. activate cm2

for (( n=0; n<=${#LIST_OF_FOLDERS[@]}; n++ ))
do
    echo "$n"    
    echo "${LIST_OF_FOLDERS[n]}"
    echo "${LIST_OF_DESTINATIONS[n]}"
    OUT0=$(sbatch --array=0 slurm_files/cm2_prep.sh "${LIST_OF_FOLDERS[n]}" "${LIST_OF_DESTINATIONS[n]}" "smartspim")
    echo "$OUT0"
    sbatch --dependency=afterany:${OUT0##* } --array=0 slurm_files/cm2_process.sh "${LIST_OF_FOLDERS[n]}" "${LIST_OF_DESTINATIONS[n]}" "smartspim"
done






