#!/bin/env bash
#
#SBATCH -c 12                      # number of cores
#SBATCH -t 720
#SBATCH -o logs/smartspim_reg_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/smartspim_reg_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 80000

sbatch spim_register.sh 0 $1 $2 $3
#sbatch spim_register.sh 1 $1 $2 $3
#sbatch spim_register.sh 2 $1 $2 $3
sbatch spim_register.sh 3 $1 $2 $3

