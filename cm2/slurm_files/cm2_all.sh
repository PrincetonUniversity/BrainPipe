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
FOLDER="$1"
echo "$FOLDER"
echo "two is "
DEST="$2"
echo "$DEST"
echo "three is "

if [[ $3 = "lavision" ]]
then
    echo "lavision"
    SCOPE="lavision"
else
    echo "smartspim"
    SCOPE="smartspim"
fi

#convert z planes to stitched npy

echo "dest and scope ins"
echo "$DEST"
echo "$SCOPE"

# add step 1
OUT2=$(sbatch --array=0-500 -p Brody slurm_files/cm2_step1.sh "$DEST" "$SCOPE")
echo $OUT2
echo "done with stp1"

# add step 3
OUT3=$(sbatch --dependency=afterany:${OUT2##* } -p Brody --array=0 slurm_files/cm2_step3.sh $DEST $SCOPE)
echo $OUT3
echo "done with stp3"




