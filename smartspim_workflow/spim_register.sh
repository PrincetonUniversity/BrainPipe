#!/bin/env bash
#
#SBATCH -c 12                      # number of cores
#SBATCH -t 600
#SBATCH -o /scratch/ejdennis/logs/smartspim_reg_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/ejdennis/logs/smartspim_reg_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 80000

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/2020.11
module load elastix/4.8
. activate lightsheet

python spim_register.py 1 "/jukebox/LightSheetData/lightserv/pbibawi/pb_udisco_647_488_4x/pb_udisco_647_488_4x-009/imaging_request_1/rawdata/resolution_3.6x/"

#functionality
#takes 3 command line arguments max
#stepid = int(sys.argv[1]) - regular or inverse transform, mostly just need inverse for cells/atlas
#src = str(sys.argv[2]) - folder to stitched images, e.g. /jukebox/LightSheetTransfer/tp/20200701_12_55_28_20170207_db_bl6_crii_rpv_01/
