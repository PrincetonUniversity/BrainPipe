#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=5000 #5 gbs
#SBATCH -t 10                # time (minutes)
#SBATCH -o val_%j.out
#SBATCH -e val_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda/5.2.0
. activate 3dunet_py3
python run_fwd.py 20181009_zd_train RSUNet 995000 20170116_tp_bl6_lob7_ml_08_647_010na_z7d5um_150msec_10povlp_ch00_C00_440-475_01 20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_C00_300-375_03 20170115_tp_bl6_lob6a_1000r_647_010na_z7d5um_125msec_10povlp_ch00_C00_300-375_01 20170116_tp_bl6_lob45_500r_12_647_010na_z7d5um_150msec_10povlp_ch00_C00_600-635_00 20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y2050-2400_x3100-3450 20170204_tp_bl6_cri_1000r_02_1hfds_647_0010na_25msec_z7d5um_10povlap_ch00_z200-400_y3800-4150_x2400-2750 20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00_z200-400_y4150-4500_x3450-3800 20170116_tp_bl6_lob7_500r_09_647_010na_z7d5um_75msec_10povlp_ch00_z200-400_y4500-4850_x3450-3800 --gpus 0 --noeval --tag noeval
