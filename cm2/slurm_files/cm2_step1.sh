#!/bin/env bash
#
#SBATCH -p Brody                # partition (queue)
#SBATCH -c 12                      # number of cores
#SBATCH -t 10                  # time (minutes)
#SBATCH -o logs/clearmap2_%j_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j_%a.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000 #120 gbs

module load anacondapy/5.3.1
. activate cm2

#make into blocks
#run cell detect on blocks
sleep $[ ( $RANDOM % 30 )  + 1 ]s
xvfb-run python cell_detect.py 1 ${FOLDER_TO_USE}
