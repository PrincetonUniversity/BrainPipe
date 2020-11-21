#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 3                      # number of cores
#SBATCH -t 800
#SBATCH -o /scratch/zmd/logs/ts_compute_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/ts_compute_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 85000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

terastitcher --displcompute --projin="$1"/xml_import.xml --subvoldim=100 --projout=xml_displcomp

