#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=5000 #5 gbs
#SBATCH -t 10                # time (minutes)
#SBATCH -o demo_%j.out
#SBATCH -e demo_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anacondapy/2020.11
. activate brainpipe

echo 'Folder to save: '
demo_folder=$(pwd)$'/demo'
echo $demo_folder
python setup_demo_script.py $demo_folder

cd pytorchutils
echo $(pwd)
python demo.py demo models/RSUNet.py samplers/demo_sampler.py augmentors/flip_rotate.py 10 --batch_sz 1 --nobn --noeval --tag demo
