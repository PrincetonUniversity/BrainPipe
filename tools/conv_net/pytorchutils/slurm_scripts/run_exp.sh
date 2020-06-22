#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --contiguous
#SBATCH --mem=20000 #20gbs
#SBATCH -t 8500                 # time (minutes)
#SBATCH -o /scratch/gpfs/zmd/logs/cnn_train.out
#SBATCH -e /scratch/gpfs/zmd/logs/cnn_train.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet
python run_exp.py 20200316_peterb_zd_train models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --batch_sz 4 --gpus 0

