#!/bin/env bash
#
#SBATCH -c 1                      # number of cores
#SBATCH -t 360                # time (minutes)
#SBATCH -o logs/cm2_filter_%j.out        # STDOUT
#SBATCH -e logs/cm2_filter_%j.err        # STDERR

module load anacondapy/2020.11
module load elastix/4.8
. activate cm2
xvfb-run python cm2_spock_filter.py
