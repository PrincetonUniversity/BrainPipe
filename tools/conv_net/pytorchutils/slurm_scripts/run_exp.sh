#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --contiguous
#SBATCH --mem=20000 #20gbs
#SBATCH -t 25                 # time (minutes)
#SBATCH -o /tigress/ahoag/cnn/exp2/slurm_logs/cnn_train_%j.out
#SBATCH -e /tigress/ahoag/cnn/exp2/slurm_logs/cnn_train_%j.err
module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/2020.11
. activate brainpipe
python run_exp.py exp2 /tigress/ahoag/cnn/exp2 models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --max_iter 201 --batch_sz 2 --chkpt_num 0 --chkpt_intv 50 --gpus 0,1
