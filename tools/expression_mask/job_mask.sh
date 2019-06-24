#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 10                      # number of cores
#SBATCH -t 360                 # time (minutes)
#SBATCH -o logs/%j.out        # STDOUT
#SBATCH -e logs/%j.err        # STDERR
#SBATCH --mail-type=FAIL      # notifications for job done & fail
#SBATCH --mail-user=deverett@princeton.edu # send-to address
#SBATCH --mem=100000      #in MB

module load anacondapy/2.7
. activate py2

#DISPLAY=localhost:10.0
#echo $DISPLAY
#echo "Suggested to try to prevent: cannot connect to X server localhost:10 as error" 

# HOW TO USE:
# submit --array=0-60 job_mask.sh 

xvfb-run --auto-servernum --server-num=1 --wait=10 python mask.py $SLURM_ARRAY_TASK_ID
#python mask.py $SLURM_ARRAY_TASK_ID
