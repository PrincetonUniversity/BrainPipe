#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=20000 #25gbs
#SBATCH -t 50                # time (minutes)
#SBATCH -o test_chnk_fwd_%j.out
#SBATCH -e test_chnk_fwd_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1
. activate 3dunet_py3
python run_chnk_fwd.py 20181009_zd_train RSUNet 995000 0 --gpus 0 --noeval --tag noeval
