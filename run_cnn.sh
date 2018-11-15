#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=20000 #20gbs
#SBATCH -t 360                # time (minutes)
#SBATCH -o chnk_%a.out
#SBATCH -e chnk_%a.err

echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1
. activate 3dunet_py3
python run_chnk_fwd.py 20181009_zd_train RSUNet 995000 ${SLURM_ARRAY_TASK} --gpus 0 --noeval --tag noeval
