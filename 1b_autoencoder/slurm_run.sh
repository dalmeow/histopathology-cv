#!/bin/bash
#SBATCH --job-name=ae_run
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --output=1b_autoencoder/logs/%j_%x.out
#SBATCH --error=1b_autoencoder/logs/%j_%x.err

# Usage: sbatch slurm_run.sh all --config ae_v1_rand

set -e

cd ~/cv/histopathology-cv/1b_autoencoder

source ~/cv/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Args:   $@"

nvidia-smi

LOG_NAME=$(echo "$@" | tr ' ' '_' | sed 's/--//g')

python run.py "$@" 2>&1 | tee "logs/${LOG_NAME}.log"
