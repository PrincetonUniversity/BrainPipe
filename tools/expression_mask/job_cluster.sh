#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 10                      # number of cores
#SBATCH -t 360                 # time (minutes)
#SBATCH -o logs/%N.%j.out        # STDOUT
#SBATCH -e logs/%N.%j.err        # STDERR
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mem=100000      #in MB

module load anacondapy/2.7
. activate py2

#DISPLAY=localhost:10.0
#echo $DISPLAY
#echo "Suggested to try to prevent: cannot connect to X server localhost:10 as error" 

# HOW TO USE:
# submit --array=0-60 job_cluster.sh 

xvfb-run --auto-servernum --server-num=1 python cluster_masks.py $SLURM_ARRAY_TASK_ID
