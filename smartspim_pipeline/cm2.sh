#!/bin/env bash
#
#SBATCH -p Brody                # partition (queue)
#SBATCH -c 12                      # number of cores
#SBATCH -t 12:00:00
#SBATCH -o /scratch/ejdennis/logs/smartspim_downsize_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/logs/smartspim_downsize_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000

module load anacondapy/5.3.1
. activate lightsheet

python CellMap_custom.py
