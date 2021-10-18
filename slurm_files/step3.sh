#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 14                 # number of cores
#SBATCH -t 700                 # time (minutes)
#SBATCH -o logs/step3_%a.out        # STDOUT
#SBATCH -e logs/step3_%a.err        # STDERR
#SBATCH --contiguous #used to try and get cpu mem to be contigous


module load anacondapy/2020.11
module load elastix/4.8
. activate brainpipe

xvfb-run python main.py 3 ${SLURM_ARRAY_TASK_ID} #run elastix; -d flag is NECESSARY for depth coding

# HOW TO USE:
# sbatch --array=0-20 sub_arrayjob.sh 
#sbatch --mail-type=END,FAIL      # notifications for job done & fail
#sbatch --mail-user=email@domain.edu # send-to address

