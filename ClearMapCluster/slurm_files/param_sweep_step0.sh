#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -n 12                      # number of cores
#SBATCH -t 500                 # time (minutes)
#SBATCH -o /scratch/zmd/logs/param_sweep_step0.out        # STDOUT
#SBATCH -e /scratch/zmd/logs/param_sweep_step0.err        # STDERR

module load anacondapy/5.3.1
module load elastix/4.8
. activate idisco

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

xvfb-run python run_parameter_sweep.py 0

# HOW TO USE:
# sbatch --array=0-20 parameter_sweep.sh
