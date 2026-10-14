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

# Run the 192-run master orchestrator on the assigned SLURM GPU
python3 run_matrix.py \
    --gpu_id="${CUDA_VISIBLE_DEVICES:-0}" \
    --output_csv="results_matrix.csv" \
    --global_epoch=20 \
    --local_epoch=5 \
    --batch_size=32 \
    --lr=0.01 \
    --resume

echo "=== Matrix run completed. Results written to results_matrix.csv ==="
