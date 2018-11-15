#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 90                # time (minutes)
#SBATCH -o /jukebox/scratch/zmd/logs/reconst_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /jukebox/scratch/zmd/logs/reconst_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 25000 #25 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.1.0
. activate lightsheet

python cell_count.py /jukebox/wang/pisano/tracing_output/antero_4x/20170308_tp_bl6_lob8_ml_04 /jukebox/scratch/zmd/20170308_tp_bl6_lob8_ml_04 

