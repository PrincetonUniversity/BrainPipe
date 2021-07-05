#!/bin/env bash
#
#SBATCH -c 12                      # number of cores
#SBATCH -t 600
#SBATCH -o logs/smartspim_reg_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/smartspim_reg_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/2020.11
module load elastix/4.8
. activate lightsheet

python spim_register.py $1 $2 $3 $4
