#!/bin/bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=5000 #5 gbs
#SBATCH -t 10                # time (minutes)
#SBATCH -o /logs/val_%j.out
#SBATCH -e /logs/val_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate lightsheet

python run_fwd.py 20200622_ed_train models/RSUNet.py 5000 z266stackstart_250_ejd --gpus 0 --noeval --tag noeval
