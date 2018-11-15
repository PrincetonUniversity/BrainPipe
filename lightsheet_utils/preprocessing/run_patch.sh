#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 8                      # number of cores
#SBATCH -t 200                # time (minutes)
#SBATCH -o /jukebox/scratch/zmd/logs/witten_m9309_patch_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /jukebox/scratch/zmd/logs/witten_m9309_patch_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 25000 #25 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.1.0
. activate lightsheet

python run_patch.py /jukebox/LightSheetData/witten-mouse/201810_RetroCre_mCherry_NAc/M9309

