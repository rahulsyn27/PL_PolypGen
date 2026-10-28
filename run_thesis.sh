#!/bin/bash
#SBATCH --job-name=PFL_Matrix
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=matrix_output.txt
#SBATCH --error=matrix_error.txt

# Activate the correct PLenv environment
source /home/rahulr/PL_PolypGen/PLenv/bin/activate
cd /home/rahulr/PL_PolypGen/
python3 run_matrix.py --gpu_id=0 --global_epoch=20 --local_epoch=5 --batch_size=32
