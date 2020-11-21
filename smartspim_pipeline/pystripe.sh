#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 12                 # number of cores
#SBATCH -t 200                 # number of minutes 
#SBATCH -o /scratch/zmd/logs/spim_pystripe_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/spim_pystripe_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 25000                      #RAM (MBs)- 25GBS

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

#required
module load anacondapy/5.3.1
. activate lightsheet

echo "Input directory (path to stitched images):" "$1"
echo "Path to flat.tiff file generated using 'Generate Flat' software:" "$2"
echo "Output directory (does not need to exist):" "$3"

pystripe -i "$1" -f "$2" -o "$3"
