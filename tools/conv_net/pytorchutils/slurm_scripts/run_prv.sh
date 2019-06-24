#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --contiguous
#SBATCH --mem=14000 #14gbs
#SBATCH -t 8500                 # time (minutes)
#SBATCH -o cnn_train_20190502.out
#SBATCH -e cnn_train_20190502.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet
python run_prv_all_inputs_exp.py 20190502_zd_transfer_learning_all_inputs models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --batch_sz 18 --chkpt_num 411740 --gpus 0,1,2,3
