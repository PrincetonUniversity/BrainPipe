#!/bin/env bash
#
#SBATCH -p Brody
#SBATCH -c 12                      # number of cores
#SBATCH -t 4                  # time (minutes)
#SBATCH -o logs/clearmap2_%j_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j_%a.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000 #120 gbs

module load anacondapy/2020.11
. activate cm2

echo "rename slurm"
echo "$1"
echo "$2"
xvfb-run python rename_smartspimZ_for_cm2.py $1 $2
