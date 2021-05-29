#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
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

module load anacondapy/5.3.1
module load elastix/4.8
. activate lightsheet

python spim_register.py 1 "$1" "$2" "Ex_488_Em_0" "Ex_647_Em_2" "rat" "jukebox/brody/lightsheet/atlasdir/mPRA.tif"

# sys.argvs are:
# 1: 0 or 1, regular or inverse
# 2: src
# 3: reg
# 4: savepath within /scratch/ejdennis/spimout
# 5: folder for cell channel e.g. Ex_642_Em_2
# 6: folder for atl, defaults to PRA

print(sys.argv)
src = str(sys.argv[2])  # folder to main image folder
reg = str(sys.argv[3])  # folder fo registration channel, e.g. Ex_488_Em_0
try:
    cell = str(sys.argv[5])  # folder for cell channel e.g. Ex_642_Em_2
except:
    cell = False
try:
    species = str(sys.argv[6])  # species to know for registration parameters
    param_fld = "/scratch/ejdennis/rat_registration_parameter_folder"  # change if using mouse
except:
    print('nope')
try:
    atl = str(sys.argv[7])
except:
    atl = "/jukebox/brody/ejdennis/lightsheet/PRA.tif"  # defaults to pra

if stepid == 0:
    svpth = os.path.join("/scratch/ejdennis/spimout", sys.argv[4])

