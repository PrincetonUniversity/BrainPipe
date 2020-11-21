#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 12                      # number of cores
#SBATCH -t 10
#SBATCH -o /scratch/zmd/logs/smartspim_downsize_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/smartspim_downsize_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 80000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.3.1
. activate lightsheet

echo "Storage directory:" "$1"

python spim_downsize.py "$1" 
