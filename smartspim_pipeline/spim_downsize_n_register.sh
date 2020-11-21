#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o /scratch/zmd/logs/spim_downsize_n_register_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/spim_downsize_n_register_%j.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

#specifications
cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/5.3.1
. activate lightsheet

echo "Storage directory:" "$1"
echo "Registration channel directory:" "$2"
echo "Cell channel directory:" "$3"

#downsize
OUT0=$(sbatch spim_downsize.sh "$1")
echo $OUT0

#register 
OUT1=$(sbatch --dependency=afterok:${OUT0##* } spim_register.sh 0 "$1" "$2" "$3")
echo $OUT1

#register inverse
OUT2=$(sbatch --dependency=afterok:${OUT0##* } spim_register.sh 1 "$1" "$2" "$3")
echo $OUT2

#functionality
#go to smartspim_pipeline folder and type sbatch spim_downsize_n_register.sh [path to main image folder] [reg channel subfolder name] [cell channel subfolder name]


