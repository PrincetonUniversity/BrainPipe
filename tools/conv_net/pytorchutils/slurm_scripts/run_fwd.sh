#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=5000 #5 gbs
#SBATCH -t 10                # time (minutes)
#SBATCH -o /tigress/ahoag/cnn/exp2/slurm_logs/cnn_inf_%j.out
#SBATCH -e /tigress/ahoag/cnn/exp2/slurm_logs/cnn_inf_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/2020.11
. activate brainpipe

python run_fwd.py exp2 /tigress/ahoag/cnn/exp2 models/RSUNet.py 12000 --gpus 0 --noeval --tag exp2
