#!/bin/bash
#SBATCH -p all
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --contiguous
#SBATCH --mem=20000 #20gbs
#SBATCH -t 8500                 # time (minutes)
#SBATCH -o /scratch/gpfs/ejdennis/logs/cnn_train.out
#SBATCH -e /scratch/gpfs/ejdennis/logs/cnn_train.err

echo $pwd

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet

python /tigress/ejdennis/BrainPipe/tools/conv_net/pytorchutils/run_exp.py 20200622_ed_train /tigress/ejdennis/BrainPipe/tools/conv_net/pytorchutils/models/RSUNet.py /tigress/ejdennis/BrainPipe/tools/conv_net/pytorchutils/samplers/soma.py /tigress/ejdennis/BrainPipe/tools/conv_net/pytorchutils/augmentors/flip_rotate.py --batch_sz 4 --gpus 0
