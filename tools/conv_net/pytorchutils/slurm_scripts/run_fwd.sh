#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=5000 #5 gbs
#SBATCH -t 10                # time (minutes)
#SBATCH -o /scratch/gpfs/ejdennis/logs/val_%j.out
#SBATCH -e /scratch/gpfs/ejdennis/logs/val_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet

python run_fwd.py 20200622_ed_train models/RSUNet.py 99000 z266stackstart_250_ejd --gpus 0 --noeval --tag noeval

