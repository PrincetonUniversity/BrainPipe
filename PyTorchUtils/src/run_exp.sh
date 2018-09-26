#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=10000 #10gbs
#SBATCH -t 1000                 # time (minutes)
#SBATCH -o logs/cnn_train_%a.out
#SBATCH -e logs/cnn_train_%a.err

module load anaconda/5.2.0
. activate 3dunet
python run_exp.py zd_training sampler RSUNet --batch_sz 32 --gpus 0

