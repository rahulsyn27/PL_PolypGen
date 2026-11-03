#!/bin/bash
#SBATCH --job-name=pfl_polypgen_matrix
#SBATCH --output=logs/matrix_%j.out
#SBATCH --error=logs/matrix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

mkdir -p logs

echo "=== SLURM Job ID: $SLURM_JOB_ID ==="
echo "=== Hostname: $(hostname) ==="
echo "=== CUDA Device: $CUDA_VISIBLE_DEVICES ==="

# Activate your python virtualenv or conda environment
# source /path/to/conda/bin/activate pfl_env

# Run the 96-run master orchestrator on the assigned SLURM GPU
# (it checks the data first; results go to results_polypgen.csv, per-run logs to logs/polypgen/)
python3 run_matrix.py \
    --gpu_id="${CUDA_VISIBLE_DEVICES:-0}" \
    --dataset=polypgen \
    --data_dir=data/PolypGen \
    --global_epoch=20 \
    --local_epoch=5 \
    --batch_size=32 \
    --lr=0.01 \
    --resume

echo "=== Matrix run completed. Results written to results_polypgen.csv ==="
