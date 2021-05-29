#!/bin/env bash
#
#SBATCH -c 11                      # number of cores
#SBATCH -t 3600                # time (minutes)
#SBATCH	--mem 100000 
#SBATCH -o /scratch/ejdennis/logs/pttransform_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/logs/pttransform_%j.err        # STDERR #add _%a to see each array job

module load elastix/4.8
module load anacondapy/2020.11
. activate lightsheet

python transform_points_group.py
