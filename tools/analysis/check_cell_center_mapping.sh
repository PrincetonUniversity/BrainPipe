#!/bin/env bash
#
#SBATCH -c 4                      # number of cores
#SBATCH -t 25                # time (minutes)
#SBATCH --mem 48G
#SBATCH -o /scratch/ejdennis/logs/cnn_eval_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/logs/cnn_eval_%a.err        # STDERR #add _%a to see each array job


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "


module load anacondapy/5.3.1
. activate lightsheet

python check_cell_center_mapping.py ${SLURM_ARRAY_TASK_ID}
