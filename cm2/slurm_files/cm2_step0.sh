#!/bin/env bash
#
#SBATCH -c 12                      # number of cores
#SBATCH -t 240                  # time (minutes)
#SBATCH -o logs/clearmap2_%j_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j_%a.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000 #120 gbs

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/2020.11
. activate cm2

#convert z planes to stitched npy
echo "STEP0 SLURM"
echo "$1"
echo "$2"
xvfb-run python cell_detect.py 0 $1 $2
