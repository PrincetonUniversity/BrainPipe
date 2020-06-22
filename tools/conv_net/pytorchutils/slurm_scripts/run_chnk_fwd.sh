#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=30000 #30gbs
#SBATCH -t 360                # time (minutes)
#SBATCH -o /scratch/gpfs/zmd/logs/array_jobs/chnk_%a_%j.out
#SBATCH -e /scratch/gpfs/zmd/logs/array_jobs/chnk_%a_%j.err

echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.3.1
. activate 3dunet

cd pytorchutils/
python run_chnk_fwd.py 20200316_peterb_zd_train  models/RSUNet.py 12000 z265 --gpus 0 --noeval --tag noeval ${SLURM_ARRAY_TASK_ID}
