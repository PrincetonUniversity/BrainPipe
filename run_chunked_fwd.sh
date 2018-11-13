#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=25000 #25gbs
#SBATCH -t 500                # time (minutes)
#SBATCH -o test_chunked_fwd_%j.out
#SBATCH -e test_chunked_fwd_%j.err

module load cudatoolkit/9.2 cudnn/cuda-9.2/7.1.4 anaconda/5.2.0
. activate 3dunet_py3
python run_chunked_fwd.py 20181009_zd_train RSUNet 995000 20180327_jg40_bl6_sim_03 --gpus 0 --noeval --tag noeval
