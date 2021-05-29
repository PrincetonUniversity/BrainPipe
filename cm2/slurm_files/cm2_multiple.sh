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
declare -a LIST_OF_FOLDERS=("/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_A296/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected" 
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_A296/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected" 
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_A300/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_A300/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected" 
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_E131/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_/pb_udisco_647_488_E131/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_E130/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488/pb_udisco_647_488_E130/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_M122/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_M122/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X073/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X073/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X077/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X077/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X078/imaging_request_1/rawdata/resolution_3.6x/Ex_488_Em_0_corrected"
"/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco/pb_udisco_X078/imaging_request_1/rawdata/resolution_3.6x/Ex_642_Em_2_corrected")

declare -a LIST_OF_DESTINATIONS=("/scratch/ejdennis/cm2_brains/A296/488" 
"/scratch/ejdennis/cm2_brains/A296/642"
"/scratch/ejdennis/cm2_brains/A300/488"
"/scratch/ejdennis/cm2_brains/A300/642"
"/scratch/ejdennis/cm2_brains/E131/488"
"/scratch/ejdennis/cm2_brains/E131/642"
"/scratch/ejdennis/cm2_brains/E130/488"
"/scratch/ejdennis/cm2_brains/E130/642"
"/scratch/ejdennis/cm2_brains/M122/488"
"/scratch/ejdennis/cm2_brains/M122/642"
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
    OUT0=$(sbatch --array=0 slurm_files/cm2_prep.sh "${LIST_OF_FOLDERS[n]}" "${LIST_OF_DESTINATIONS[n]}" "smartspim")
    echo "$OUT0"
    sbatch --dependency=afterany:${OUT0##* } --array=0 slurm_files/cm2_process.sh "${LIST_OF_FOLDERS[n]}" "${LIST_OF_DESTINATIONS[n]}" "smartspim"
done






