#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=30000 #30gbs
#SBATCH -t 360                # time (minutes)
#SBATCH -o /tigress/ahoag/cnn/exp2/slurm_logs/array_jobs/chnk_%a_%j.out
#SBATCH -e /tigress/ahoag/cnn/exp2/slurm_logs/array_jobs/chnk_%a_%j.err

echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet

cd pytorchutils/
python run_chnk_fwd.py exp2 /tigress/ahoag/cnn/exp2  models/RSUNet.py 12000 --gpus 0 --noeval --tag exp2 ${SLURM_ARRAY_TASK_ID}
