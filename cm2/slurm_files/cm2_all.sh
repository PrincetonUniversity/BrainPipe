#!/bin/env bash
#

#SBATCH -c 12                      # number of cores
#SBATCH -t 4                  # time (minutes)
#SBATCH -o logs/clearmap2_%j_%a.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/clearmap2_%j_%a.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 120000 #120 gbs

PYTHONPATH="${PYTHONPATH}:/scratch/ejdennis/rat_BrainPipe/ClearMap2"

module load anacondapy/2020.11
. activate cm2

echo "one is "
echo "$1"
echo "two is "
echo "$2"

if [[ $3 = "lavision" ]]
then
    echo "lavision"
    SCOPE="smartspim"
else
    echo "smartspim"
    SCOPE="smartspim"
    xvfb-run python rename_smartspimZ_for_cm2.py $1 $2
fi

#convert z planes to stitched npy
xvfb-run python cell_detect.py 0 $2 $SCOPE

#make into blocks
#run cell detect on blocks
sleep $[ ( $RANDOM % 30 )  + 1 ]s
xvfb-run python cell_detect.py 1 $2 $SCOPE

#combine blocks
xvfb-run python cell_detect.py 3 $2 $SCOPE






