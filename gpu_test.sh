#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8000
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --output='gpu_test_%a.log'

module load anaconda/5.2.0
module load cudatoolkit/9.2
module load cudnn/cuda-9.2/7.1.4

which python
which nvcc
echo $CUDNN_ROOT
python gpu_test.py
~                                                                               
~                                                                               
~                                                                               
~                                                                               
~  
