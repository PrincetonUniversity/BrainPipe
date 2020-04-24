#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --contiguous
#SBATCH --mem=10000 #10gbs
#SBATCH -t 900                 # time (minutes)
#SBATCH -o cnn_train_cfos.out
#SBATCH -e cnn_train_cfos.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet

python run_cfos.py 20190607_zd_transfer_learning models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --batch_sz 500 --chkpt_num 321500 --gpus 0,1,2,3
