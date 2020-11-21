#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 3                      # number of cores
#SBATCH -t 40
#SBATCH -o /scratch/zmd/logs/ts_proj_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/ts_proj_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 5000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

terastitcher --displproj --projin="$1"/xml_displcomp.xml

terastitcher --displthres --projin="$1"/xml_displproj.xml --projout="$1"/xml_displthres.xml --threshold=0.7

terastitcher --placetiles --projin="$1"/xml_displthres.xml --projout="$1"/xml_placetiles.xml --algorithm=MST
