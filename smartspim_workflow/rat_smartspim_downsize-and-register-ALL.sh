#!/bin/env bash
#
#SBATCH -p Brody                # partition (queue)
#SBATCH -c 11                      # number of cores
#SBATCH -t 10:00:00                # time (minutes)
#SBATCH -o /scratch/ejdennis/logs/spim_downsize_n_register_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/logs/spim_downsize_n_register_%j.err        # STDERR #add _%a to see each array job
#SBATCH --mem=128000
echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

#specifications
cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/2020.11
. activate lightsheet

module load elastix/4.8 

# should be able to provide a PROJECT folder like this /jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/
# and a SAVE folder /scratch/ejdennis/projectout
# then index into project folder using array

echo "input folder:" "$1"
echo "save folder:" "$2"

#downsize
python spim_downsize_and_register.py "$1" "$2"
