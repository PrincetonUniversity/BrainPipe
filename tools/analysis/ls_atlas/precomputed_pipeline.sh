#!/bin/env bash
#
# --- PURPOSE ---
# Pipeline to make precomputed (i.e. Neuroglancer-friendly) 
# volumes for the MRI rat atlas 

# author: Austin Hoag
# date: 04/08/2020


# First raw data
echo "Jobids for raw data precomputed steps 0, 1, 2"

#Step 0: Make info file and layer directory
# OUT0=$(sbatch --parsable --export=ALL precomputed_atlas_step0.sh) 
# echo $OUT0

#Step 1: Upload raw data to vol (writes precomputed data to disk)

# OUT1=$(sbatch --parsable --dependency=afterok:${OUT0##* } precomputed_atlas_step1.sh) 
OUT1=$(sbatch --parsable precomputed_atlas_step1.sh) 
echo $OUT1

#Step 2: Downsample 

# OUT1=$(sbatch --parsable --dependency=afterok:${OUT0##* } precomputed_atlas_step1.sh) 
# OUT2=$(sbatch --parsable precomputed_atlas_step2.sh) 
# echo $OUT2


# Usage notes:
# after = go once the specified job starts
# afterany = go if the specified job finishes, regardless of success
# afternotok = go if the specified job fails
# afterok = go if the specified job completes successfully
