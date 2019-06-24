#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                    # number of cores
#SBATCH -t 20                # time (minutes)
#SBATCH -o /scratch/zmd/logs/cnn_step4_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/cnn_step4_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 5000 #5gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.3.1
. activate lightsheet

echo "Experiment name:" "`pwd`"

python cell_detect.py 4 0 "`pwd`" 
