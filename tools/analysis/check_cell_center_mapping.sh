#!/bin/env bash
#
#SBATCH -c 1                      # number of cores
#SBATCH -t 1                # time (minutes)
#SBATCH -o /scratch/ejdennis/cnn_eval_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/cnn_eval_%j.err        # STDERR #add _%a to see each array job


echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

